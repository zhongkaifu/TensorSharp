// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TSGgml_HostReadBarrier drains GPU work an op left queued (Metal async compute). It used to
// clear its flag and then sync, so a second caller arriving mid-drain returned at once, before
// the work it had to wait for had completed. The drain now holds a lock across the sync and a
// caller arriving mid-drain waits for it; the native test tests/host_read_gate_test.cpp shows
// that with a sync it blocks, and that a queued result is drained exactly once however many
// threads ask. This test pins, through the shipped library, the fact that design leans on for
// CUDA: it never queues one, so its callers never reach the lock (a rank thread held there behind
// a sync that waits for its half of an in-kernel collective would deadlock). The only op that
// still queues a result is a graph's on Metal with async compute.
using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using TensorSharp.GGML;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>The barrier's flag is process-wide: an op on any other thread can raise or drain it.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class GgmlHostReadBarrierCollection
{
    public const string Name = "GGML host read barrier";
}

[Collection(GgmlHostReadBarrierCollection.Name)]
public sealed class GgmlHostReadBarrierTests
{
    private const int Heads = 4, KvHeads = 2, HeadDim = 64, BlockSize = 4, Keys = 8, Queries = 2;
    private const int Rows = Queries * Heads * HeadDim;
    private const int Callers = 8;

    [DllImport("GgmlOps", CallingConvention = CallingConvention.Cdecl)]
    private static extern int TSGgml_HostReadBarrier();

    /// <summary>
    /// The paged attention on CUDA downloads its result before returning (it cannot bind a host
    /// pointer), so nothing is left for the barrier.
    /// </summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void HostReadBarrier_IsNeverArmedByCudaPagedAttention()
    {
        _ = new GgmlContext([0], GgmlBackendType.Cuda);
        var output = new float[Rows];
        Assert.Equal((0, 0), Drain(() => PagedAttentionHost(output)));
        Assert.All(output, x => Assert.True(float.IsFinite(x)));
    }

    /// <summary>
    /// Empties the barrier, runs <paramref name="op"/>, then has <see cref="Callers"/> threads
    /// call the barrier at once. Returns how many of them synced the backend, and whether one
    /// more call after them did.
    /// </summary>
    private static (int Concurrent, int After) Drain(Action op)
    {
        while (TSGgml_HostReadBarrier() != 0) { }
        op();
        using var start = new Barrier(Callers);
        int drained = 0;
        var threads = Enumerable.Range(0, Callers).Select(_ => new Thread(() =>
        {
            start.SignalAndWait();
            Interlocked.Add(ref drained, TSGgml_HostReadBarrier());
        })).ToArray();
        foreach (var thread in threads) thread.Start();
        foreach (var thread in threads)
            Assert.True(thread.Join(TimeSpan.FromSeconds(30)), "a barrier caller hung");
        return (drained, TSGgml_HostReadBarrier());
    }

    private static void PagedAttentionHost(float[] output)
    {
        var (q, k, v) = Inputs();
        GgmlBasicOps.PagedAttentionForward(q, k, v, output, new[] { 0, Queries }, new[] { Keys },
            new[] { Keys - 2, Keys - 1 }, Enumerable.Range(0, Keys / BlockSize).ToArray(), new[] { 0 },
            1, Queries, Heads, KvHeads, HeadDim, BlockSize, 1f / MathF.Sqrt(HeadDim));
    }

    private static (float[] q, float[] k, float[] v) Inputs()
    {
        var random = new Random(5);
        float[] q = new float[Rows];
        float[] k = new float[Keys * KvHeads * HeadDim];
        float[] v = new float[k.Length];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(random.NextDouble() - 0.5);
        for (int i = 0; i < k.Length; i++) { k[i] = (float)(random.NextDouble() - 0.5); v[i] = (float)(random.NextDouble() - 0.5); }
        return (q, k, v);
    }
}
