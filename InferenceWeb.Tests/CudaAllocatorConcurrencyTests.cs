// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Validates the lock-free CudaAllocator device-memory pool:
//   1. Correctness + metric accounting under highly concurrent Rent/Return.
//   2. A baseline-vs-new microbenchmark proving the per-size-class
//      ConcurrentStack design scales better than the original single global
//      `poolSync` monitor it replaced.
//
// The microbenchmark uses a fake (free) backing allocator so it isolates the
// managed pool's synchronization cost — exactly the hot spot the change
// targets — and therefore runs without a GPU. The real-device correctness and
// throughput tests are gated on CudaBackend.IsAvailable().
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class CudaAllocatorConcurrencyTests
{
    private readonly ITestOutputHelper _output;
    private static long s_sink;

    public CudaAllocatorConcurrencyTests(ITestOutputHelper output) => _output = output;

    // ---------------------------------------------------------------------
    // Real-device correctness: concurrent allocate/dispose must be race-free,
    // must reuse pooled blocks (high hit ratio), and must keep cachedBytes /
    // metric accounting consistent (no negative or leaked accounting).
    // ---------------------------------------------------------------------
    [Fact]
    public void RealDevice_ConcurrentAllocateDispose_IsRaceFreeAndPools()
    {
        if (!CudaBackend.IsAvailable())
        {
            _output.WriteLine("[cuda-alloc] CUDA unavailable; skipping real-device test.");
            return;
        }

        using var allocator = new CudaAllocator();

        const int threads = 16;
        const int opsPerThread = 4000;
        // A small set of repeated shapes so the steady state is dominated by
        // pool hits (the contention-sensitive path).
        long[] elementCounts = { 64, 256, 1024, 4096, 16384 };

        var failures = new ConcurrentQueue<Exception>();
        long checksum = 0;
        RunParallel(threads, t =>
        {
            var rng = new Random(1000 + t);
            long localSum = 0;
            for (int i = 0; i < opsPerThread; i++)
            {
                long n = elementCounts[rng.Next(elementCounts.Length)];
                try
                {
                    using var tensor = new Tensor(allocator, DType.Float32, n);
                    localSum += tensor.ElementCount();
                }
                catch (Exception ex)
                {
                    failures.Enqueue(ex);
                    return;
                }
            }
            Interlocked.Add(ref checksum, localSum);
        });

        Assert.True(failures.IsEmpty, failures.IsEmpty ? "" : $"Allocation threw: {failures.TryPeek(out var e)} {e}");

        CudaAllocatorStats stats = allocator.GetStats();
        _output.WriteLine($"[cuda-alloc] {stats}");

        long totalRents = threads * (long)opsPerThread;
        Assert.Equal(totalRents, stats.TotalRentCount);
        Assert.True(stats.PoolHitCount + stats.PoolMissCount == totalRents);
        // Every distinct size class only needs to be cuMemAlloc'd a bounded
        // number of times (≈ peak concurrent live blocks of that size), so the
        // overwhelming majority of rents are served from the pool.
        Assert.True(stats.PoolHitRatio > 0.9,
            $"Expected high pool reuse, got hitRatio={stats.PoolHitRatio:P2} (alloc={stats.CuMemAllocCount})");
        // Accounting invariants: never negative, cap respected.
        Assert.True(stats.CachedBytes >= 0, $"cachedBytes went negative: {stats.CachedBytes}");
        Assert.True(stats.CachedBytes <= stats.MaxCachedBytes, $"cachedBytes {stats.CachedBytes} exceeded cap {stats.MaxCachedBytes}");
        Assert.True(stats.PeakCachedBytes >= stats.CachedBytes);
    }

    // ---------------------------------------------------------------------
    // Deterministic single-threaded metric accounting.
    // ---------------------------------------------------------------------
    [Fact]
    public void RealDevice_Metrics_CountHitsMissesAndDriverAllocs()
    {
        if (!CudaBackend.IsAvailable())
        {
            _output.WriteLine("[cuda-alloc] CUDA unavailable; skipping metrics test.");
            return;
        }

        using var allocator = new CudaAllocator();
        CudaAllocatorStats start = allocator.GetStats();

        // First allocation of a fresh size: a miss + one cuMemAlloc.
        var t1 = new Tensor(allocator, DType.Float32, 7777);
        CudaAllocatorStats afterAlloc = allocator.GetStats();
        Assert.Equal(start.PoolMissCount + 1, afterAlloc.PoolMissCount);
        Assert.Equal(start.CuMemAllocCount + 1, afterAlloc.CuMemAllocCount);

        // Dispose returns the block to the pool (no cuMemFree).
        t1.Dispose();
        CudaAllocatorStats afterReturn = allocator.GetStats();
        Assert.Equal(afterAlloc.ReturnedToPoolCount + 1, afterReturn.ReturnedToPoolCount);
        Assert.Equal(afterAlloc.CuMemFreeCount, afterReturn.CuMemFreeCount);
        Assert.True(afterReturn.CachedBytes > afterAlloc.CachedBytes);

        // Re-allocating the same rounded size is a pool hit with no new driver alloc.
        using var t2 = new Tensor(allocator, DType.Float32, 7777);
        CudaAllocatorStats afterReuse = allocator.GetStats();
        Assert.Equal(afterReturn.PoolHitCount + 1, afterReuse.PoolHitCount);
        Assert.Equal(afterReturn.CuMemAllocCount, afterReuse.CuMemAllocCount);
    }

    // ---------------------------------------------------------------------
    // Real-device throughput (informational): aggregate alloc/free ops/sec
    // across thread counts. Never fails on throughput; just prints.
    // ---------------------------------------------------------------------
    [Fact]
    public void RealDevice_AllocationThroughput_ScalesWithThreads()
    {
        if (!CudaBackend.IsAvailable())
        {
            _output.WriteLine("[cuda-alloc] CUDA unavailable; skipping throughput test.");
            return;
        }

        using var allocator = new CudaAllocator();
        long[] elementCounts = { 256, 1024, 4096 };
        const int opsPerThread = 3000;

        _output.WriteLine("[cuda-alloc] real-device CudaAllocator allocate+dispose throughput");
        _output.WriteLine($"{"threads",7} {"ops",9} {"wall(ms)",9} {"ops/sec",12}");
        foreach (int threads in new[] { 1, 2, 4, 8, 16 })
        {
            var sw = Stopwatch.StartNew();
            RunParallel(threads, t =>
            {
                var rng = new Random(7 + t);
                for (int i = 0; i < opsPerThread; i++)
                {
                    long n = elementCounts[rng.Next(elementCounts.Length)];
                    using var tensor = new Tensor(allocator, DType.Float32, n);
                }
            });
            sw.Stop();
            long ops = threads * (long)opsPerThread;
            double opsPerSec = ops / sw.Elapsed.TotalSeconds;
            _output.WriteLine($"{threads,7} {ops,9} {sw.Elapsed.TotalMilliseconds,9:F1} {opsPerSec,12:N0}");
        }

        _output.WriteLine($"[cuda-alloc] final {allocator.GetStats()}");
    }

    // ---------------------------------------------------------------------
    // Baseline (global lock) vs new (per-size-class ConcurrentStack) pool.
    // Same fake backing allocator + identical size-rounding; the only
    // difference is the synchronization strategy. Proves the new design wins
    // under concurrency. GPU-independent.
    // ---------------------------------------------------------------------
    [Fact]
    public void ShardedPool_OutperformsGlobalLock_UnderHighConcurrency()
    {
        const int opsPerThread = 300_000;
        long[] sizes = { 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 };
        int[] threadCounts = { 1, 2, 4, 8, Math.Max(8, Environment.ProcessorCount) };

        using var probe = new RealPoolAdapter();
        _output.WriteLine($"[pool-bench] cores={Environment.ProcessorCount}, opsPerThread={opsPerThread:N0}, " +
                          $"sizeClasses={sizes.Length}, poolShards={probe.ShardCount}");
        _output.WriteLine($"{"threads",7} {"baseline(ops/s)",16} {"striped(ops/s)",16} {"speedup",8}");

        double maxThreadSpeedup = 0;
        int maxThreads = 0;
        foreach (int threads in threadCounts)
        {
            // Each pool is freshly warmed inside MeasurePool so the steady state is
            // pool-hit dominated. The "striped" pool is the real CudaDeviceMemoryPool
            // wired to a free fake backing allocator, so this measures the actual
            // production pool code against a faithful replica of the old global-lock
            // design — isolating the synchronization strategy.
            double baseOps = MeasurePool(new GlobalLockPool(), threads, opsPerThread, sizes);
            using var striped = new RealPoolAdapter();
            double stripedOps = MeasurePool(striped, threads, opsPerThread, sizes);
            double speedup = baseOps > 0 ? stripedOps / baseOps : 0;
            _output.WriteLine($"{threads,7} {baseOps,16:N0} {stripedOps,16:N0} {speedup,7:F2}x");
            if (threads >= maxThreads)
            {
                maxThreads = threads;
                maxThreadSpeedup = speedup;
            }
        }

        // At the highest concurrency the striped pool must beat the global-lock
        // baseline: the baseline serializes every rent and return on one monitor,
        // while the striped pool routes each core to an almost-private shard.
        Assert.True(maxThreadSpeedup >= 1.1,
            $"Striped pool did not beat global lock at {maxThreads} threads: speedup={maxThreadSpeedup:F2}x");
        _output.WriteLine($"[pool-bench] speedup at {maxThreads} threads: {maxThreadSpeedup:F2}x");
    }

    // ---- helpers ----------------------------------------------------------

    private static void RunParallel(int threads, Action<int> body)
    {
        var workers = new Thread[threads];
        using var barrier = new Barrier(threads);
        var error = new ConcurrentQueue<Exception>();
        for (int t = 0; t < threads; t++)
        {
            int id = t;
            workers[t] = new Thread(() =>
            {
                try
                {
                    barrier.SignalAndWait();
                    body(id);
                }
                catch (Exception ex)
                {
                    error.Enqueue(ex);
                }
            }) { IsBackground = true };
            workers[t].Start();
        }
        foreach (var w in workers)
            w.Join();
        if (!error.IsEmpty && error.TryDequeue(out var first))
            throw first;
    }

    private static double MeasurePool(IBenchPool pool, int threads, int opsPerThread, long[] sizes)
    {
        // Warm-up: seed each size class so steady-state rents hit the pool.
        var seeded = new IntPtr[sizes.Length];
        for (int i = 0; i < sizes.Length; i++)
            seeded[i] = pool.Rent(sizes[i], out _);
        for (int i = 0; i < sizes.Length; i++)
            pool.Return(seeded[i], RoundAllocationSize(sizes[i]));

        long checksum = 0;
        var sw = Stopwatch.StartNew();
        RunParallelLocal(threads, t =>
        {
            long local = 0;
            int idx = t;
            for (int i = 0; i < opsPerThread; i++)
            {
                long size = sizes[(idx + i) & (sizes.Length - 1)];
                IntPtr p = pool.Rent(size, out long alloc);
                local += p.ToInt64();
                pool.Return(p, alloc);
            }
            Interlocked.Add(ref checksum, local);
        });
        sw.Stop();
        // Publish checksum to a sink so the rent/return loop cannot be optimized away.
        Volatile.Write(ref s_sink, checksum);
        double ops = threads * (double)opsPerThread;
        return ops / sw.Elapsed.TotalSeconds;
    }

    private static void RunParallelLocal(int threads, Action<int> body)
    {
        var workers = new Thread[threads];
        using var barrier = new Barrier(threads);
        for (int t = 0; t < threads; t++)
        {
            int id = t;
            workers[t] = new Thread(() => { barrier.SignalAndWait(); body(id); }) { IsBackground = true };
            workers[t].Start();
        }
        foreach (var w in workers)
            w.Join();
    }

    // Identical rounding to CudaAllocator.RoundAllocationSize so both bench
    // pools place blocks into the same size classes the real allocator uses.
    private static long RoundAllocationSize(long bytes)
    {
        const long smallMax = 1L << 20;
        if (bytes <= 256) return 256;
        if (bytes <= smallMax)
        {
            long size = 256;
            while (size < bytes) size <<= 1;
            return size;
        }
        const long largeAlignment = 1L << 20;
        return ((bytes + largeAlignment - 1) / largeAlignment) * largeAlignment;
    }

    private interface IBenchPool
    {
        IntPtr Rent(long requestedBytes, out long allocationBytes);
        void Return(IntPtr ptr, long allocationBytes);
    }

    // Mirrors the ORIGINAL CudaAllocator pool: one global monitor guarding a
    // Dictionary<long, Stack<IntPtr>>; cachedBytes mutated under the lock.
    private sealed class GlobalLockPool : IBenchPool
    {
        private readonly object poolSync = new object();
        private readonly Dictionary<long, Stack<IntPtr>> devicePool = new Dictionary<long, Stack<IntPtr>>();
        private readonly long maxCachedBytes = 512L * 1024 * 1024;
        private readonly long maxCachedBlockBytes = 256L * 1024 * 1024;
        private long cachedBytes;
        private long nextHandle;

        public IntPtr Rent(long requestedBytes, out long allocationBytes)
        {
            allocationBytes = RoundAllocationSize(Math.Max(requestedBytes, 1));
            lock (poolSync)
            {
                if (devicePool.TryGetValue(allocationBytes, out Stack<IntPtr> stack) && stack.Count > 0)
                {
                    cachedBytes -= allocationBytes;
                    return stack.Pop();
                }
            }
            return new IntPtr(Interlocked.Increment(ref nextHandle));
        }

        public void Return(IntPtr ptr, long allocationBytes)
        {
            if (ptr == IntPtr.Zero) return;
            if (allocationBytes > 0 && allocationBytes <= maxCachedBlockBytes)
            {
                lock (poolSync)
                {
                    if (cachedBytes + allocationBytes <= maxCachedBytes)
                    {
                        if (!devicePool.TryGetValue(allocationBytes, out Stack<IntPtr> stack))
                        {
                            stack = new Stack<IntPtr>();
                            devicePool[allocationBytes] = stack;
                        }
                        stack.Push(ptr);
                        cachedBytes += allocationBytes;
                    }
                }
            }
        }
    }

    // Wraps the REAL production CudaDeviceMemoryPool (striped per-core shards,
    // array-backed Stack, per-shard byte counters) over a free fake backing
    // allocator so the benchmark measures the actual shipped pool code.
    private sealed class RealPoolAdapter : IBenchPool, IDisposable
    {
        private readonly CudaDeviceMemoryPool _pool;
        private long _counter;

        public RealPoolAdapter()
        {
            _pool = new CudaDeviceMemoryPool(
                maxCachedBytes: 512L * 1024 * 1024,
                maxCachedBlockBytes: 256L * 1024 * 1024,
                enabled: true,
                backingAllocate: _ => new IntPtr(Interlocked.Increment(ref _counter)),
                backingFree: _ => { });
        }

        public int ShardCount => _pool.ShardCount;

        public IntPtr Rent(long requestedBytes, out long allocationBytes) => _pool.Rent(requestedBytes, out allocationBytes);

        public void Return(IntPtr ptr, long allocationBytes) => _pool.Return(ptr, allocationBytes);

        public void Dispose() => _pool.DrainAndFree();
    }
}
