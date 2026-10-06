// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;

namespace InferenceWeb.Tests;

public sealed class Qwen4ExpTensorParallelTests
{
    [Theory]
    [InlineData(BackendType.GgmlCpu, 7)]
    [InlineData(BackendType.GgmlCuda, 7)]
    [InlineData(BackendType.GgmlMetal, int.MaxValue)]
    [InlineData(BackendType.GgmlVulkan, int.MaxValue)]
    public void SpeculativeDraftLimitLeavesRoomForTheVerifyAnchor(BackendType backend, int expected)
        => Assert.Equal(expected, Qwen4ExpModel.ResolveSpecMaxDraftTokens(backend));

    [Fact]
    public void ArchitectureKeepsTensorAndLayerParallelismDistinct()
    {
        Assert.True(ModelArchitectureRegistry.TryGet("qwen4exp", out var architecture));
        Assert.Equal(MultiGpuMode.TensorParallel, architecture.MultiGpu);
        Assert.True(architecture.SupportsLayerSplit);
        Assert.False(architecture.SupportsDistributedTensorParallel);
        ITensorParallelGroup group = null!;
        Assert.Equal(2, ModelBase.ResolveTensorParallelSupport(architecture, BackendType.GgmlCuda, 2, ref group, out int split));
        Assert.Equal(1, split);
        Assert.Equal(1, ModelBase.ResolveTensorParallelSupport(architecture, BackendType.GgmlCuda, 1, ref group, out split, 2));
        Assert.Equal(2, split);
        Assert.Throws<ArgumentException>(() => ModelBase.ResolveTensorParallelSupport(architecture, BackendType.GgmlCuda, 2, ref group, out _, 2));
    }

    [Theory]
    [InlineData(0, false, 8, 12)] // F32 gate/up
    [InlineData(0, true, 12, 8)]  // F32 down
    [InlineData(2, false, 64, 12)] // Q4_0 gate/up
    [InlineData(2, true, 640, 8)] // Actual model FFN width, quant-block-aligned TP2
    [InlineData(14, true, 512, 8)] // Q6_K down, complete 256-element blocks
    [InlineData(13, false, 256, 12)] // Q5_K gate/up
    [InlineData(16, false, 256, 12)] // IQ2_XXS gate/up
    [InlineData(17, false, 2560, 640)] // Target model IQ2_XS gate/up; keep its 2560-element dots intact
    [InlineData(18, false, 2560, 640)] // Target model IQ3_XXS gate/up
    [InlineData(19, false, 256, 12)] // IQ1_S gate/up
    [InlineData(22, false, 256, 12)] // IQ2_S gate/up
    public void ExpertShardsPartitionEverySourceByte(int type, bool rowParallel, int ne0, int ne1)
    {
        const int degree = 2, experts = 3;
        long rowBytes = NativeDequant.RowSize(type, ne0);
        int sourceBytes = checked((int)(rowBytes * ne1 * experts));
        byte[] source = Enumerable.Range(0, sourceBytes).Select(i => (byte)((i * 73 + i / 251) % 256)).ToArray();
        byte[] reconstructed = new byte[sourceBytes];
        IntPtr data = QuantizedWeight.AllocateBuffer(sourceBytes);
        Marshal.Copy(source, 0, data, sourceBytes);
        var weight = new StackedExpertWeights(data, type, ne0, ne1, experts, sourceBytes, true, source, IntPtr.Zero);
        try
        {
            for (int rank = 0; rank < degree; ++rank)
            {
                var shard = Qwen4ExpModel.SliceTensorParallelExpert(weight, rank, degree, rowParallel);
                try
                {
                    Assert.Equal(experts, shard.NumExperts);
                    Assert.Equal(sourceBytes / degree, shard.TotalRawBytes);
                    Assert.Equal(rowParallel ? ne0 / degree : ne0, shard.PerExpertNe0);
                    Assert.Equal(rowParallel ? ne1 : ne1 / degree, shard.PerExpertNe1);
                    byte[] bytes = new byte[shard.TotalRawBytes];
                    Marshal.Copy(shard.Data, bytes, 0, bytes.Length);
                    int localExpert = checked((int)shard.PerExpertRawBytes);
                    int originalExpert = sourceBytes / experts;
                    int localRow = rowParallel ? (int)rowBytes / degree : (int)rowBytes;
                    for (int e = 0; e < experts; ++e)
                        if (rowParallel)
                            for (int row = 0; row < ne1; ++row)
                                Array.Copy(bytes, e * localExpert + row * localRow, reconstructed,
                                    e * originalExpert + row * (int)rowBytes + rank * localRow, localRow);
                        else
                            Array.Copy(bytes, e * localExpert, reconstructed, e * originalExpert + rank * localExpert, localExpert);
                }
                finally { QuantizedWeight.FreeBuffer(shard.OwnedBuffer); }
            }
            Assert.Equal(source, reconstructed);
        }
        finally { QuantizedWeight.FreeBuffer(data); }
    }

    [Theory]
    [InlineData(13, 2)]
    [InlineData(13, 4)]
    [InlineData(16, 2)]
    [InlineData(16, 4)]
    [InlineData(17, 2)]
    [InlineData(17, 4)]
    [InlineData(18, 2)]
    [InlineData(18, 4)]
    [InlineData(19, 2)]
    [InlineData(19, 4)]
    [InlineData(21, 2)]
    [InlineData(21, 4)]
    [InlineData(22, 2)]
    [InlineData(22, 4)]
    public void MmqEdgeTilesRetainExactSourceRowsAndCoverEveryLogicalChannel(int type, int degree)
    {
        const int input = 256, output = 640, experts = 3;
        int alignment = Qwen4ExpModel.Qwen4ExpMmqRowAlignment(type, output);
        Assert.Equal(128, alignment);
        long row = NativeDequant.RowSize(type, input);
        byte[] bytes = Enumerable.Range(0, checked((int)(row * output * experts)))
            .Select(i => (byte)((i * 37 + i / 113) % 256)).ToArray();
        IntPtr data = QuantizedWeight.AllocateBuffer(bytes.Length);
        Marshal.Copy(bytes, 0, data, bytes.Length);
        var source = new StackedExpertWeights(data, type, input, output, experts, bytes.Length, true, bytes, IntPtr.Zero);
        try
        {
            for (int rank = 0; rank < degree; ++rank)
            {
                var range = Qwen4ExpModel.Qwen4ExpOutputRowRange(output, rank, degree, alignment);
                Assert.Equal(0, range.First % 128);
                Assert.Equal(0, range.Count % 128);
                Assert.InRange(rank * output / degree - range.First, 0, 127);
                Assert.True(range.First + range.Count >= (rank + 1) * output / degree);
                var shard = Qwen4ExpModel.SliceTensorParallelExpert(source, rank, degree, false, alignment);
                try
                {
                    Assert.Equal(input, shard.PerExpertNe0);
                    Assert.Equal(range.Count, shard.PerExpertNe1);
                    byte[] actual = new byte[shard.TotalRawBytes];
                    Marshal.Copy(shard.Data, actual, 0, actual.Length);
                    for (int expert = 0; expert < experts; ++expert)
                        Assert.Equal(bytes.AsSpan(checked((int)((expert * output + range.First) * row)),
                            checked((int)(range.Count * row))).ToArray(),
                            actual.AsSpan(checked((int)(expert * range.Count * row)), checked((int)(range.Count * row))).ToArray());
                }
                finally { QuantizedWeight.FreeBuffer(shard.OwnedBuffer); }
            }
        }
        finally { QuantizedWeight.FreeBuffer(data); }
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void DownOutputShardsPreserveTheEntireDotProductWidth(int degree)
    {
        const int type = 8, input = 640, output = 2560;
        long bytes = NativeDequant.RowSize(type, input) * output;
        IntPtr data = QuantizedWeight.AllocateBuffer(bytes);
        var weight = new StackedExpertWeights(data, type, input, output, 1, bytes, true, null!, IntPtr.Zero);
        try
        {
            for (int rank = 0; rank < degree; ++rank)
            {
                var shard = Qwen4ExpModel.SliceTensorParallelExpert(weight, rank, degree, false);
                try
                {
                    Assert.Equal(input, shard.PerExpertNe0);
                    Assert.Equal(output / degree, shard.PerExpertNe1);
                    Assert.Equal(bytes / degree, shard.TotalRawBytes);
                }
                finally { QuantizedWeight.FreeBuffer(shard.OwnedBuffer); }
            }
        }
        finally { QuantizedWeight.FreeBuffer(data); }
    }

    [Theory]
    [InlineData(0, false, 8, 5, 2)] // Output tail cannot be dropped.
    [InlineData(2, true, 96, 8, 2)] // 48 is not a whole Q4_0 block.
    [InlineData(14, true, 768, 8, 2)] // 384 is not a whole Q6_K block.
    [InlineData(17, false, 640, 8, 2)] // Output-row slicing still needs complete source IQ2_XS blocks.
    [InlineData(17, true, 768, 8, 2)] // Input slices cannot split a 256-element IQ2_XS block.
    public void UnsupportedSlicesFailBeforeReadingOrAllocating(int type, bool rowParallel, int ne0, int ne1, int degree)
    {
        var weight = new StackedExpertWeights(IntPtr.Zero, type, ne0, ne1, 3, 0, true, null!, IntPtr.Zero);
        Assert.Throws<NotSupportedException>(() => Qwen4ExpModel.SliceTensorParallelExpert(weight, 0, degree, rowParallel));
    }

    [Theory]
    [InlineData(29, 640)] // IQ1_M has no upstream MMQ kernel.
    [InlineData(17, 608)] // Source output rows must form whole 128-row MMQ tiles.
    public void UnsupportedMmqLayoutsDoNotRequestOverlappingTiles(int type, int rows)
        => Assert.Equal(1, Qwen4ExpModel.Qwen4ExpMmqRowAlignment(type, rows));
}
