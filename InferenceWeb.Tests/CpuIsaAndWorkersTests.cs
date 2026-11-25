// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading;
using TensorSharp.Cpu;
using TensorSharp.Models.QwenImage;
using QGemmIsa = TensorSharp.Models.ManagedQuantizedOps.QGemmIsa;

namespace InferenceWeb.Tests;

/// <summary>
/// The one ISA decision (CpuIsa) and the one fork/join helper (CpuWorkers) the managed CPU
/// kernels share. Every kernel family must pick AVX-512 exactly when CpuIsa says so, so one
/// host never mixes kernel widths (the per-row dots that predate CpuIsa keep their own test, see
/// below); and a CpuWorkers set must run every block exactly once and surface failures, whichever
/// of the pool or Parallel.For backs it.
/// </summary>
[Collection("CPU kernel selection")]
public class CpuIsaAndWorkersTests
{
    [Fact]
    public void EveryKernelFamilyFollowsTheSharedIsaDecision()
    {
        // The flags imply each other the way the kernels assume.
        if (CpuIsa.HasAvx512) Assert.True(CpuIsa.HasAvx2Fma);
        Assert.Equal(CpuIsa.HasAvx512 && !CpuIsa.Avx512DisabledByEnv, CpuIsa.Avx512);
        Assert.Equal(CpuIsa.HasAvx2Fma, CpuIsa.Avx2Fma);

        Assert.Equal(CpuIsa.Avx512 ? CpuGemmIsa.Avx512 : CpuIsa.Avx2Fma ? CpuGemmIsa.Avx2 : CpuGemmIsa.Portable,
            CpuPackedGemm.DetectIsa());
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_CPU_SGEMM_KERNEL")))
            Assert.Equal(CpuIsa.Avx512 ? CpuSgemm.KernelKind.Avx512 : CpuIsa.Avx2Fma ? CpuSgemm.KernelKind.Avx2 : CpuSgemm.KernelKind.Portable,
                CpuSgemm.DefaultKernel());
        QGemmIsa q = ManagedQuantizedOps.ResolveQGemmIsa(QGemmIsa.Auto);
        if (q != QGemmIsa.PerRow)   // PerRow: no AVX2
            Assert.Equal(CpuIsa.Avx512 ? QGemmIsa.Avx512 : QGemmIsa.Avx2, q);
        else
            Assert.False(CpuIsa.Avx2Fma);
    }

    public static TheoryData<string> WorkerSets() => new() { "pool", "threadpool", "shared", "wide" };

    private static CpuWorkers Make(string kind, CpuWorkerPool pool) => kind switch
    {
        "pool" => CpuWorkers.On(pool),
        "threadpool" => CpuWorkers.OnThreadPool(3),
        "shared" => CpuWorkers.Shared,
        _ => CpuPackedGemm.WidePool,
    };

    [Theory]
    [MemberData(nameof(WorkerSets))]
    public void ForRunsEveryIndexOnceAndRethrows(string kind)
    {
        using var pool = new CpuWorkerPool(3);
        CpuWorkers workers = Make(kind, pool);
        Assert.True(workers.ThreadCount >= 1);
        if (kind == "pool") Assert.Equal(3, workers.ThreadCount);
        if (kind == "threadpool") Assert.Equal(3, workers.ThreadCount);

        foreach (int count in new[] { 0, 1, 2, 7, 1000 })
        {
            var hits = new int[count];
            workers.For(count, i => Interlocked.Increment(ref hits[i]));
            Assert.All(hits, h => Assert.Equal(1, h));
        }

        var error = Assert.ThrowsAny<Exception>(() => workers.For(64, i =>
        {
            if (i == 17) throw new InvalidOperationException("block 17");
        }));
        var inner = error is AggregateException aggregate ? aggregate.Flatten().InnerExceptions[0] : error;
        Assert.IsType<InvalidOperationException>(inner);
        Assert.Equal("block 17", inner.Message);
    }

    [Fact]
    public void TheThreadPoolFallbackKeepsItsWidthCap()
    {
        int concurrent = 0, peak = 0;
        CpuWorkers.OnThreadPool(2).For(64, _ =>
        {
            int now = Interlocked.Increment(ref concurrent);
            int seen;
            while (now > (seen = Volatile.Read(ref peak)) && Interlocked.CompareExchange(ref peak, now, seen) != seen) { }
            Thread.Sleep(1);
            Interlocked.Decrement(ref concurrent);
        });
        Assert.InRange(peak, 1, 2);
    }
}
