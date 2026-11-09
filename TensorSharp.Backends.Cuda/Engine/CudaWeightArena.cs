// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Weight residency for the whole-model direct-CUDA engines (DeepSeek V4, GLM-5.3-Flash):
// every packed quantized weight a device holds lives in one allocator-owned block, the
// device's arena, and is streamed into it from the GGUF shards by a bounded pool of
// reader threads.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Random-access byte source an engine streams weights from during load. Lets a
    /// multi-hundred-GB model go GGUF shard -> pinned chunk -> VRAM without ever staging
    /// the whole thing in host RAM.
    /// </summary>
    /// <remarks>
    /// Implementations are called concurrently from every loader thread and must be
    /// thread-safe (positional reads, no shared file cursor).
    /// </remarks>
    public interface ICudaWeightSource
    {
        /// <summary>Reads exactly <paramref name="bytes"/> bytes at <paramref name="offset"/>
        /// into <paramref name="dst"/>, or throws.</summary>
        void Read(long offset, IntPtr dst, long bytes);
    }

    /// <summary>
    /// Optional capability of an <see cref="ICudaWeightSource"/>: hand out a stable read-only
    /// pointer into a file mapping of the shard. Weights that stay in host RAM for the engine's
    /// lifetime (the <c>--n-cpu-moe</c> routed experts) borrow the mapping instead of taking a
    /// private copy: file-backed pages are page cache the kernel can evict and re-read, so
    /// experts that outweigh the host's memory allowance (the cgroup limit on containers,
    /// where a private 137 GiB copy is a silent OOM SIGKILL) degrade to storage speed instead
    /// of killing the process.
    /// </summary>
    public interface ICudaMappedWeightSource : ICudaWeightSource
    {
        /// <summary>
        /// Maps <c>[offset, offset+bytes)</c> of the shard read-only. The pointer stays valid
        /// until the source is disposed, which therefore must not happen before the engine that
        /// borrowed it. False when the platform or filesystem cannot map; callers fall back to
        /// a copy.
        /// </summary>
        bool TryMapRange(long offset, long bytes, out IntPtr ptr);
    }

    /// <summary>A packed ggml-quantized weight as the loader describes it: where its bytes come
    /// from and the shape they encode.</summary>
    public struct CudaWeightDesc
    {
        /// <summary>Host staging pointer. Zero when the weight streams straight from
        /// <see cref="Source"/> into VRAM.</summary>
        public IntPtr HostPtr;
        /// <summary>Byte source the upload reads from when <see cref="HostPtr"/> is zero (the
        /// GGUF shard, normally).</summary>
        public ICudaWeightSource Source;
        public long SourceOffset;
        public int GgmlType;
        public int Ne0;
        public int Ne1;
        public int Ne2;
        public long RowBytes;
        public string Name;
        public bool IsValid => HostPtr != IntPtr.Zero || Source != null;
        public long TotalBytes => (long)Ne1 * Math.Max(1, Ne2) * RowBytes;
    }

    /// <summary>A weight resident in a device arena: its address and the shape the matmul
    /// routing needs.</summary>
    internal struct DeviceWeight
    {
        public IntPtr Ptr;
        public int Type;
        public int Ne0;
        public int Ne1;
        public long RowBytes;
        /// <summary>An engine's optional F16 copy for prefill-size GEMMs (zero when it has none).</summary>
        public IntPtr PrefillF16;
        /// <summary>A stacked expert weight whose experts do not all sit in one device block: a device
        /// table of one address per expert (a VRAM slot or pinned host memory the GPU reads over the
        /// bus). Zero means <see cref="Ptr"/> is the whole stack.</summary>
        public IntPtr Table;
    }

    /// <summary>
    /// One device's weight arena. <see cref="Place"/> bump-allocates a weight's bytes (a single
    /// block keeps thousands of weight tensors from each paying an allocation-granularity tax)
    /// and either copies them from host staging at once or plans a streamed read, which
    /// <see cref="StreamAll"/> then runs for every device with the reader concurrency the
    /// filesystem likes.
    /// </summary>
    internal sealed class CudaWeightArena : IDisposable
    {
        private sealed class UploadJob
        {
            public IntPtr Dst;
            public ICudaWeightSource Src;
            public long SrcOffset;
            public long Bytes;
        }

        // The read side is the bottleneck on network filesystems and its throughput is not
        // monotonic in thread count -- MooseFS/FUSE peaks around 8-32 concurrent readers and
        // falls off a cliff above that (2.4 GB/s at 16 threads vs 1.0 GB/s at 96), so the pool
        // is explicitly bounded rather than left to the thread pool's discretion.
        private static readonly int LoaderThreads = Math.Max(1, EnvInt("TS_DSV4_LOAD_THREADS", 16));
        private static readonly int LoaderChunkBytes = Math.Max(1 << 20, EnvInt("TS_DSV4_LOAD_CHUNK_MB", 16) << 20);
        private const long UploadSegmentBytes = 128L << 20;
        private static readonly bool LoaderStats = EnvInt("TS_DSV4_LOAD_STATS", 0) != 0;
        private static long _loaderReadTicks;
        private static long _loaderCopyTicks;

        private readonly CudaAllocator _alloc;
        private readonly Tensor _block;
        private readonly IntPtr _base;
        private List<UploadJob> _plan = new List<UploadJob>();
        private UploadJob[] _jobs;
        private int _cursor;

        public long Bytes { get; }
        public long Used { get; private set; }

        public CudaWeightArena(CudaAllocator alloc, long bytes)
        {
            _alloc = alloc;
            Bytes = Math.Max(bytes, 256);
            _block = new Tensor(alloc, DType.UInt8, Bytes);
            _base = ((CudaStorage)_block.Storage).DevicePtrAtElement(_block.StorageOffset);
        }

        public static long Align(long v) => (v + 255) & ~255L;

        /// <summary>Reserve <paramref name="bytes"/> (256-byte aligned) of the arena.</summary>
        public IntPtr Take(long bytes)
        {
            long aligned = Align(bytes);
            if (Used + aligned > Bytes)
                throw new InvalidOperationException($"[cuda-engine] weight arena overflow on device {_alloc.DeviceId}");
            IntPtr p = (IntPtr)((long)_base + Used);
            Used += aligned;
            return p;
        }

        /// <summary>Reserve <paramref name="bytes"/> of uninitialized arena memory; returns the device address.</summary>
        public IntPtr Reserve(long bytes) => Take(bytes);

        /// <summary>Copy <paramref name="bytes"/> from host memory into the arena now; returns the device
        /// address. The device's context must be current.</summary>
        public IntPtr PlaceBytes(IntPtr host, long bytes)
        {
            IntPtr p = Take(bytes);
            CudaDriverApi.cuMemcpyHtoD(p, host, new UIntPtr((ulong)bytes)).ThrowOnError();
            return p;
        }

        /// <summary>Place a weight; an invalid descriptor yields a default (absent) weight. The
        /// device's context must be current.</summary>
        public DeviceWeight Place(in CudaWeightDesc qw)
        {
            if (!qw.IsValid)
                return default;
            long bytes = qw.TotalBytes;
            IntPtr p = Take(bytes);
            if (qw.HostPtr != IntPtr.Zero)
            {
                CudaDriverApi.cuMemcpyHtoD(p, qw.HostPtr, new UIntPtr((ulong)bytes)).ThrowOnError();
            }
            else
            {
                // Planned now, transferred by StreamAll so the (slow) reads run wide instead of
                // one tensor at a time. Split into segments so several loader threads can share
                // one multi-hundred-MB expert stack.
                for (long off = 0; off < bytes; off += UploadSegmentBytes)
                {
                    _plan.Add(new UploadJob
                    {
                        Dst = (IntPtr)((long)p + off),
                        Src = qw.Source,
                        SrcOffset = qw.SourceOffset + off,
                        Bytes = Math.Min(UploadSegmentBytes, bytes - off),
                    });
                }
            }
            return new DeviceWeight { Ptr = p, Type = qw.GgmlType, Ne0 = qw.Ne0, Ne1 = qw.Ne1, RowBytes = qw.RowBytes };
        }

        /// <summary>
        /// Streams every planned weight of every arena from its GGUF shard into VRAM: a bounded
        /// pool of reader threads per device, each double-buffering through pinned host chunks so
        /// the file read of chunk N+1 overlaps the HtoD of chunk N.
        /// </summary>
        public static void StreamAll(IReadOnlyList<CudaWeightArena> arenas, string tag)
        {
            long total = 0;
            int jobCount = 0;
            foreach (var arena in arenas)
            {
                arena._jobs = arena._plan.ToArray();
                arena._plan = new List<UploadJob>();
                arena._cursor = 0;
                jobCount += arena._jobs.Length;
                foreach (var j in arena._jobs)
                    total += j.Bytes;
            }
            if (jobCount == 0)
                return;

            var sw = Stopwatch.StartNew();
            int perDev = Math.Max(1, LoaderThreads / arenas.Count);
            var errors = new List<Exception>();
            var threads = new List<System.Threading.Thread>(perDev * arenas.Count);
            foreach (var arenaLocal in arenas)
            {
                var arena = arenaLocal;
                if (arena._jobs.Length == 0)
                    continue;
                for (int w = 0; w < perDev; w++)
                {
                    var t = new System.Threading.Thread(() =>
                    {
                        try
                        {
                            arena.StreamWorker();
                        }
                        catch (Exception ex)
                        {
                            lock (errors)
                                errors.Add(ex);
                        }
                    });
                    t.IsBackground = true;
                    threads.Add(t);
                    t.Start();
                }
            }
            foreach (var t in threads)
                t.Join();
            if (errors.Count > 0)
                throw new AggregateException($"[{tag}] weight streaming failed", errors);

            double gib = total / (1024.0 * 1024 * 1024);
            Console.Error.WriteLine($"[{tag}] streamed {gib:F1} GiB of weights into VRAM in " +
                $"{sw.Elapsed.TotalSeconds:F1}s ({gib / Math.Max(0.001, sw.Elapsed.TotalSeconds):F2} GiB/s, " +
                $"{threads.Count} readers)");
            if (LoaderStats)
            {
                double readS = System.Threading.Interlocked.Read(ref _loaderReadTicks) / (double)Stopwatch.Frequency;
                double copyS = System.Threading.Interlocked.Read(ref _loaderCopyTicks) / (double)Stopwatch.Frequency;
                Console.Error.WriteLine($"[{tag}]   thread-seconds: read {readS:F1}s " +
                    $"({gib / Math.Max(0.001, readS) * threads.Count:F2} GiB/s aggregate), copy-wait {copyS:F1}s");
            }
        }

        private void StreamWorker()
        {
            _alloc.Context.MakeCurrent();
            int chunk = LoaderChunkBytes;
            IntPtr stream = IntPtr.Zero;
            var bufs = new IntPtr[2];
            var evs = new IntPtr[2];
            var pending = new bool[2];
            try
            {
                CudaDriverApi.cuStreamCreate(out stream, 0x1 /*NON_BLOCKING*/).ThrowOnError();
                for (int i = 0; i < 2; i++)
                {
                    CudaDriverApi.cuMemHostAlloc(out bufs[i], new UIntPtr((ulong)chunk), 0x1 /*PORTABLE*/).ThrowOnError();
                    CudaDriverApi.cuEventCreate(out evs[i], 0x02 /*DISABLE_TIMING*/).ThrowOnError();
                }

                var jobs = _jobs;
                int slot = 0;
                while (true)
                {
                    int idx = System.Threading.Interlocked.Increment(ref _cursor) - 1;
                    if (idx >= jobs.Length)
                        break;
                    var job = jobs[idx];
                    for (long done = 0; done < job.Bytes; )
                    {
                        long n = Math.Min(chunk, job.Bytes - done);
                        // Reclaim the staging buffer only once its copy retired.
                        if (pending[slot])
                        {
                            long tc = LoaderStats ? Stopwatch.GetTimestamp() : 0;
                            CudaDriverApi.cuEventSynchronize(evs[slot]).ThrowOnError();
                            if (LoaderStats)
                                System.Threading.Interlocked.Add(ref _loaderCopyTicks, Stopwatch.GetTimestamp() - tc);
                            pending[slot] = false;
                        }
                        long t0 = LoaderStats ? Stopwatch.GetTimestamp() : 0;
                        job.Src.Read(job.SrcOffset + done, bufs[slot], n);
                        if (LoaderStats)
                            System.Threading.Interlocked.Add(ref _loaderReadTicks, Stopwatch.GetTimestamp() - t0);
                        CudaDriverApi.cuMemcpyHtoDAsync((IntPtr)((long)job.Dst + done), bufs[slot],
                            new UIntPtr((ulong)n), stream).ThrowOnError();
                        CudaDriverApi.cuEventRecord(evs[slot], stream).ThrowOnError();
                        pending[slot] = true;
                        done += n;
                        slot ^= 1;
                    }
                }
                CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
            }
            finally
            {
                for (int i = 0; i < 2; i++)
                {
                    if (evs[i] != IntPtr.Zero) CudaDriverApi.cuEventDestroy(evs[i]);
                    if (bufs[i] != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(bufs[i]);
                }
                if (stream != IntPtr.Zero) CudaDriverApi.cuStreamDestroy(stream);
            }
        }

        private static int EnvInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out int v) ? v : fallback;
        }

        public void Dispose() => _block.Dispose();
    }
}
