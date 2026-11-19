// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Linq;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// A recorded graph owns the device blocks its kernels work in: nothing else may rent them while it can replay.
/// </summary>
public sealed class CudaGraphCaptureOwnershipTests
{
    /// <summary>
    /// A block freed inside a concurrent region is held back until the region joins (CudaAllocator.EnterSideStream). When
    /// the join came after the capture ended, the block went back to the pool while the graph still wrote through it: a
    /// parked Qwen3.8 decode recording replayed into the half tables a later prompt had rented there (CUDA error 700).
    /// </summary>
    [CudaFact]
    public void ABlockHeldBackPastTheCaptureStaysTheGraphs()
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        const int n = 4096;
        using Tensor x = Tensor.FromArray(allocator, Enumerable.Range(0, n).Select(i => (float)i).ToArray());
        using Tensor y = new Tensor(allocator, DType.Float32, n), input = new Tensor(allocator, DType.Float32, n);
        foreach (Tensor t in new[] { x, y, input }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        using var side = CudaStream.Create();
        using var graphs = new CudaPrefillGraphCache(allocator);
        long Base(Tensor t) => ((CudaStorage)t.Storage).DeviceBuffer.ToInt64();

        Assert.True(graphs.BeginCapture("step"));
        // A region opens (its work stays on the capturing stream here) and holds returns back until it joins.
        allocator.EnterSideStream(side);
        allocator.LeaveSideStream();
        long scratchBase;
        using (var scratch = new Tensor(allocator, DType.Float32, n))
        {
            scratchBase = Base(scratch);
            Ops.Add(scratch, x, x);
            Ops.Add(y, scratch, x);
        }
        Assert.True(graphs.EndCaptureAndLaunch("step", input, new[] { y }));
        allocator.EndDeferredReturns();   // the join, after the capture
        allocator.Stream.Synchronize();

        using var other = new Tensor(allocator, DType.Float32, n);
        Assert.NotEqual(scratchBase, Base(other));
        Ops.Fill(other, -1f);
        Ops.Fill(y, 0f);
        graphs.Replay("step");
        allocator.Stream.Synchronize();
        ((CudaStorage)y.Storage).MarkDeviceModified();
        Assert.Equal(Enumerable.Range(0, n).Select(i => 3f * i).ToArray(), y.GetElementsAsFloat(n));
        Assert.All(other.GetElementsAsFloat(n), v => Assert.Equal(-1f, v));
    }

    /// <summary>
    /// A capture's work recycles blocks: one returned part way and rented again for something kept past the capture (the
    /// executor's q8_1 copies lived until its next run) was not in the pool when the capture ended, so the graph did not
    /// take it, and its holder later returned it to the pool while every kernel that had used it still wrote there (a
    /// Qwen3.8 27B DFlash2 step's delta-net q and k: acceptance fell from 44% to 19% and an output changed).
    /// </summary>
    [CudaFact]
    public void ABlockStillRentedWhenTheCaptureEndsIsTheGraphsOnceReturned()
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        const int n = 4096;
        using Tensor x = Tensor.FromArray(allocator, Enumerable.Range(0, n).Select(i => (float)i).ToArray());
        using Tensor y = new Tensor(allocator, DType.Float32, n), input = new Tensor(allocator, DType.Float32, n);
        foreach (Tensor t in new[] { x, y, input }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        using var graphs = new CudaPrefillGraphCache(allocator);
        long Base(Tensor t) => ((CudaStorage)t.Storage).DeviceBuffer.ToInt64();

        Assert.True(graphs.BeginCapture("step"));
        long scratchBase;
        using (var scratch = new Tensor(allocator, DType.Float32, n))
        {
            scratchBase = Base(scratch);
            Ops.Add(scratch, x, x);
            Ops.Add(y, scratch, x);
        }
        var kept = new Tensor(allocator, DType.Float32, n);
        Assert.Equal(scratchBase, Base(kept));   // the same block, rented again
        Ops.Copy(kept, y);
        Assert.True(graphs.EndCaptureAndLaunch("step", input, new[] { y }));
        allocator.Stream.Synchronize();
        kept.Dispose();   // its holder lets it go after the capture

        using var other = new Tensor(allocator, DType.Float32, n);
        Assert.NotEqual(scratchBase, Base(other));
        Ops.Fill(other, -1f);
        Ops.Fill(y, 0f);
        graphs.Replay("step");
        allocator.Stream.Synchronize();
        ((CudaStorage)y.Storage).MarkDeviceModified();
        Assert.Equal(Enumerable.Range(0, n).Select(i => 3f * i).ToArray(), y.GetElementsAsFloat(n));
        Assert.All(other.GetElementsAsFloat(n), v => Assert.Equal(-1f, v));
    }

    /// <summary>
    /// A block the capture's work returned stays in the pool from the capture's end until the graph takes it. The pool's
    /// pressure relief (a return under the free-memory reserve, on any thread: the GC finalizer's too) gives cached blocks
    /// back to the driver unasked; run in that window it freed the graph's scratch, the graph found it neither in the pool
    /// nor rented and claimed a freed address, which the driver then handed to the next allocation the replays wrote into.
    /// The relief runs here on the test's thread, as another thread's would, once the capture has ended.
    /// </summary>
    [CudaFact]
    public void APressureReliefAsTheCaptureEnds_LeavesTheGraphItsBlocks()
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        const int n = 1 << 20;   // 4 MiB: the pool's large blocks, which the relief gives back
        using Tensor x = Tensor.FromArray(allocator, Enumerable.Range(0, n).Select(i => (float)(i % 1000)).ToArray());
        using Tensor y = new Tensor(allocator, DType.Float32, n), input = new Tensor(allocator, DType.Float32, n);
        foreach (Tensor t in new[] { x, y, input }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        using var graphs = new CudaPrefillGraphCache(allocator);
        long Base(Tensor t) => ((CudaStorage)t.Storage).DeviceBuffer.ToInt64();
        long relievedInTheWindow = -1;
        graphs.CaptureEndedForTests = () => relievedInTheWindow = allocator.RelievePressureForTests(1L << 40);

        Assert.True(graphs.BeginCapture("step"));
        long scratchBase;
        using (var scratch = new Tensor(allocator, DType.Float32, n))
        {
            scratchBase = Base(scratch);
            Ops.Add(scratch, x, x);
            Ops.Add(y, scratch, x);
        }
        Assert.True(graphs.EndCaptureAndLaunch("step", input, new[] { y }));
        allocator.Stream.Synchronize();

        Assert.Equal(0, relievedInTheWindow);
        using var other = new Tensor(allocator, DType.Float32, n);
        Assert.NotEqual(scratchBase, Base(other));
        Ops.Fill(other, -1f);
        Ops.Fill(y, 0f);
        graphs.Replay("step");
        allocator.Stream.Synchronize();
        ((CudaStorage)y.Storage).MarkDeviceModified();
        Assert.Equal(Enumerable.Range(0, n).Select(i => 3f * (i % 1000)).ToArray(), y.GetElementsAsFloat(n));
        Assert.All(other.GetElementsAsFloat(n), v => Assert.Equal(-1f, v));

        // Once the graph holds its blocks the relief runs again.
        new Tensor(allocator, DType.Float32, n).Dispose();
        Assert.True(allocator.RelievePressureForTests(4L << 20) >= 4L << 20);
    }
}
