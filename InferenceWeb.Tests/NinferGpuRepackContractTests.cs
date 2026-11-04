// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

/// <summary>CPU geometry/range contracts only; no kernel or device qualification.</summary>
public sealed class NinferGpuRepackContractTests
{
    [Theory]
    [InlineData(128,64,CudaNinferOriginalRowMapping.Identity)]
    [InlineData(384,192,CudaNinferOriginalRowMapping.Identity)]
    [InlineData(128,65536,CudaNinferOriginalRowMapping.Identity)]
    [InlineData(34816,5120,CudaNinferOriginalRowMapping.GateUpRows8)]
    [InlineData(14336,5120,CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue)]
    public void CompleteOriginalObjectAndSeparateDestinationAreBounded(int rows,int columns,CudaNinferOriginalRowMapping mapping)
    {
        var g=CudaNinferRepackModule.ValidateGeometry(rows,columns,mapping);
        Assert.Equal((long)rows*columns/2,g.CodeBytes);Assert.Equal((long)rows*columns/16,g.ScaleBytes);
        Assert.Equal(g.CodeBytes+g.ScaleBytes,g.DestinationBytes);Assert.Equal(g.DestinationBytes+4,g.OriginalBytes);
        Assert.InRange(g.OriginalBytes,1,CudaNinferRepackModule.MaximumOriginalBytes);
        Assert.Equal((long)rows*columns/16,g.Groups);
    }
    [Theory]
    [InlineData(0,64)]
    [InlineData(127,64)]
    [InlineData(129,64)]
    [InlineData(128,0)]
    [InlineData(128,63)]
    [InlineData(128,65)]
    [InlineData(128,65600)]
    public void IncompleteOrUnboundedGeometryIsRejected(int rows,int columns)
        =>Assert.Throws<ArgumentOutOfRangeException>(()=>CudaNinferRepackModule.ValidateGeometry(rows,columns,CudaNinferOriginalRowMapping.Identity));
    [Fact]
    public void OriginalObjectCannotExceedStagingBudget()
        =>Assert.Throws<NotSupportedException>(()=>CudaNinferRepackModule.ValidateGeometry(65536,65536,CudaNinferOriginalRowMapping.Identity));
    [Fact]
    public void MappingHasItsOwnGeometryContract()
    {
        Assert.Throws<ArgumentException>(()=>CudaNinferRepackModule.ValidateGeometry(128,64,CudaNinferOriginalRowMapping.GateUpRows8));
        Assert.Throws<ArgumentException>(()=>CudaNinferRepackModule.ValidateGeometry(256,64,CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue));
        Assert.Throws<ArgumentOutOfRangeException>(()=>CudaNinferRepackModule.ValidateGeometry(128,64,(CudaNinferOriginalRowMapping)3));
    }
    [Theory]
    [InlineData(0,0x10000)]
    [InlineData(0x10000,0)]
    [InlineData(-8,0x10000)]
    [InlineData(0x10000,0x20001)]
    [InlineData(0x10001,0x20000)]
    public void InvalidOrUnalignedBufferRangesAreRejected(long source,long destination)
        =>Assert.Throws<ArgumentException>(()=>CudaNinferRepackModule.ValidateBuffers(new IntPtr(source),new IntPtr(destination),
            CudaNinferRepackModule.ValidateGeometry(128,64,CudaNinferOriginalRowMapping.Identity)));
    [Fact]
    public void OverlapAndOverflowAreRejectedBeforeDispatch()
    {
        var g=CudaNinferRepackModule.ValidateGeometry(128,64,CudaNinferOriginalRowMapping.Identity);
        Assert.Throws<ArgumentException>(()=>CudaNinferRepackModule.ValidateBuffers(new IntPtr(0x10000),new IntPtr(0x10000),g));
        Assert.Throws<ArgumentException>(()=>CudaNinferRepackModule.ValidateBuffers(new IntPtr(0x10000),new IntPtr(0x10008),g));
        // Even the four-byte original calibration trailer remains source-owned.
        Assert.Throws<ArgumentException>(()=>CudaNinferRepackModule.ValidateBuffers(new IntPtr(0x10000),new IntPtr(0x10000+g.DestinationBytes),g));
        Assert.Throws<ArgumentException>(()=>CudaNinferRepackModule.ValidateBuffers(new IntPtr(long.MaxValue-7),new IntPtr(0x10000),g));
        Assert.Throws<ArgumentException>(()=>CudaNinferRepackModule.ValidateBuffers(new IntPtr(0x10000),new IntPtr(long.MaxValue-7),g));
        CudaNinferRepackModule.ValidateBuffers(new IntPtr(0x10000),new IntPtr(0x10000+g.OriginalBytes+4),g);
    }
}
