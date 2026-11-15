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
/// A recorded graph moved to other bound buffers (CudaGraphRelocation): its kernels read and write the new buffers at the
/// same offsets, and the old ones are left as they were.
/// </summary>
public sealed class CudaGraphRelocationTests
{
    [CudaFact]
    public void ARelocatedGraphReadsAndWritesItsNewBuffers()
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        const int n = 1000;
        Tensor Make(float scale) => Tensor.FromArray(allocator, Enumerable.Range(0, n).Select(i => scale * i).ToArray());
        using Tensor x = Make(1), y = Make(0), x2 = Make(3), y2 = Make(0), input = Make(0);
        foreach (Tensor t in new[] { x, y, x2, y2, input }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        using var graphs = new CudaPrefillGraphCache(allocator) { KeepTemplates = true };
        Assert.True(graphs.BeginCapture("step"));
        // y = x + x, then its second half from its first (views: pointers at offsets inside the bound buffers).
        Ops.Add(y, x, x);
        using (Tensor head = y.Narrow(0, 0, n / 2))
        using (Tensor tail = y.Narrow(0, n / 2, n / 2))
            Ops.Add(tail, tail, head);
        Assert.True(graphs.EndCaptureAndLaunch("step", input, new[] { y }));
        allocator.Stream.Synchronize();
        ((CudaStorage)y.Storage).MarkDeviceModified();
        float[] Expected(float scale) => Enumerable.Range(0, n).Select(i => 2 * scale * i + (i >= n / 2 ? 2 * scale * (i - n / 2) : 0)).ToArray();
        Assert.Equal(Expected(1), y.GetElementsAsFloat(n));

        Assert.True(graphs.TryGetGraphs("step", out System.IntPtr exec, out System.IntPtr template));
        long Base(Tensor t) => ((CudaStorage)t.Storage).DeviceBuffer.ToInt64();
        long Bytes(Tensor t) => t.Storage.ByteLength;
        var relocation = new CudaGraphRelocation(template, new[] { (Base(x), Bytes(x)), (Base(y), Bytes(y)) });
        Assert.Null(relocation.Refusal);
        Assert.True(relocation.PatchedNodes >= 2);
        Assert.True(relocation.Apply(exec, allocator.Context.Handle, new[] { Base(x2), Base(y2) }));
        graphs.ReplaceKeepAlive("step", new[] { y2 });
        graphs.Replay("step");
        allocator.Stream.Synchronize();
        ((CudaStorage)y2.Storage).MarkDeviceModified();
        Assert.Equal(Expected(3), y2.GetElementsAsFloat(n));
        Assert.Equal(Expected(1), y.GetElementsAsFloat(n));   // the old output untouched

        // And back.
        Assert.True(relocation.Apply(exec, allocator.Context.Handle, new[] { Base(x), Base(y) }));
        Ops.Fill(y, 0);
        graphs.Replay("step");
        allocator.Stream.Synchronize();
        ((CudaStorage)y.Storage).MarkDeviceModified();
        Assert.Equal(Expected(1), y.GetElementsAsFloat(n));
    }
}
