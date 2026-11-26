// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The pure-C# backend's allocator keeps large freed buffers for the next storage of their size (a step's intermediates are
/// written in place instead of mapped again), promises a zero only for a buffer that came fresh from the C allocator, and
/// gives everything back once disposed with its model.
/// </summary>
public sealed class RecyclingCpuAllocatorTests
{
    [Fact]
    public void AFreedLargeBufferServesTheNextStorageOfItsSize_AndOnlyAFreshOneIsPromisedZero()
    {
        using var cpu = new RecyclingCpuAllocator(BlasEnum.DotNet, 1L << 30);
        long large = CpuStorage.ZeroedAllocationBytes / sizeof(float);
        IntPtr first;
        using (var t = new Tensor(cpu, DType.Float32, large))
        {
            Assert.True(t.Storage.TakeAllocationZeroGuarantee());
            first = TensorComputePrimitives.GetStoragePointer(t);
            Ops.Fill(t, 1f);
        }
        Assert.Equal(CpuStorage.ZeroedAllocationBytes, cpu.RetainedBytes);
        using (var again = new Tensor(cpu, DType.Float32, large))
        {
            Assert.Equal(first, TensorComputePrimitives.GetStoragePointer(again));
            Assert.False(again.Storage.TakeAllocationZeroGuarantee());   // it holds the last storage's values
            Assert.Equal(0, cpu.RetainedBytes);
        }
        using (var small = new Tensor(cpu, DType.Float32, RecyclingCpuAllocator.MinRecycledBytes / sizeof(float)))
            Assert.False(small.Storage.TakeAllocationZeroGuarantee());
    }

    [Fact]
    public void ADisposedAllocatorKeepsNothing_EvenFromStoragesThatOutliveIt()
    {
        var cpu = new RecyclingCpuAllocator(BlasEnum.DotNet, 1L << 30);
        var outlives = new Tensor(cpu, DType.Float32, RecyclingCpuAllocator.MinRecycledBytes);
        new Tensor(cpu, DType.Float32, RecyclingCpuAllocator.MinRecycledBytes).Dispose();
        Assert.True(cpu.RetainedBytes > 0);
        cpu.Dispose();
        Assert.Equal(0, cpu.RetainedBytes);
        outlives.Dispose();
        Assert.Equal(0, cpu.RetainedBytes);
    }

    [Fact]
    public void ThePureCSharpBackendRecyclesAndReleasesWithItsModel()
    {
        string dir = Directory.CreateTempSubdirectory("recycling-cpu").FullName;
        try
        {
            string path = new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3", LayerCount = 2 }
                .Write(Path.Combine(dir, "mistral3.gguf"));
            RecyclingCpuAllocator cpu;
            using (ModelBase model = ModelBase.Create(path, BackendType.Cpu))
            {
                cpu = Assert.IsType<RecyclingCpuAllocator>(model.DirectAllocatorForTelemetry);
                model.ForwardRefill(Enumerable.Range(0, 40).Select(i => 30 + i % 150).ToArray());
            }
            Assert.Equal(0, cpu.RetainedBytes);
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { }
        }
    }
}
