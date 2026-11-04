// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using TensorSharp.Cuda;
using Xunit;

public sealed class Qwen35NinferGdnChunkPolicyTests
{
    [Theory]
    [InlineData(1,1,0,1,0,false)]
    [InlineData(63,63,0,63,0,false)]
    [InlineData(64,64,64,0,0,true)]
    [InlineData(65,65,64,1,0,true)]
    [InlineData(127,127,64,63,0,true)]
    [InlineData(128,128,128,0,0,true)]
    [InlineData(4096,130,128,2,3966,true)]
    [InlineData(4096,17,0,17,4079,false)]
    [InlineData(4096,0,0,0,4096,false)]
    public void RealSpanControlsChunkAndTailRecipe(int width,int actual,int full,int tail,int padding,bool materialize)
    {
        var plan=Q35CudaEngine.PlanNinferGdnChunk(width,actual);
        Assert.Equal(actual,plan.ActualRows);Assert.Equal(full,plan.FullRows);
        Assert.Equal(tail,plan.TailRows);Assert.Equal(padding,plan.PaddingRows);
        Assert.Equal(materialize,plan.MaterializeBf16Qk);
        Assert.Equal(width,plan.FullRows+plan.TailRows+plan.PaddingRows);
    }
    [Theory]
    [InlineData(0,0)]
    [InlineData(-1,0)]
    [InlineData(64,-1)]
    [InlineData(64,65)]
    public void InvalidRealSpanIsRejected(int width,int actual)
        =>Assert.Throws<ArgumentOutOfRangeException>(()=>Q35CudaEngine.PlanNinferGdnChunk(width,actual));
    [Fact]
    public void UnsetModuleDoesNotDiscoverOrEnableCandidate()
        =>Assert.Null(Q35CudaEngine.ResolveNinferGdnChunkPath(null));
    [Theory]
    [InlineData("")]
    [InlineData(" ")]
    [InlineData("\t")]
    public void ExplicitBlankModuleIsRejected(string setting)
        =>Assert.Throws<ArgumentException>(()=>Q35CudaEngine.ResolveNinferGdnChunkPath(setting));
    [Fact]
    public void ExplicitMissingModuleIsRejected()
        =>Assert.Throws<FileNotFoundException>(()=>Q35CudaEngine.ResolveNinferGdnChunkPath(Path.Combine(Path.GetTempPath(),Guid.NewGuid()+".fatbin")));
    [Fact]
    public void ExplicitExistingPathIsNormalized()
    {
        string path=Path.GetTempFileName();
        try{Assert.Equal(Path.GetFullPath(path),Q35CudaEngine.ResolveNinferGdnChunkPath(path));}
        finally{File.Delete(path);}
    }
    [Fact]
    public void ExistingRelativeModulePathIsRejected()
    {
        string path=Path.GetTempFileName();
        try
        {
            string relative=Path.GetRelativePath(Environment.CurrentDirectory,path);
            Assert.False(Path.IsPathFullyQualified(relative));
            Assert.Throws<ArgumentException>(()=>Q35CudaEngine.ResolveNinferGdnChunkPath(relative));
        }
        finally{File.Delete(path);}
    }
    [Fact]
    public void ExistingDotSegmentModulePathIsRejected()
    {
        string path=Path.GetTempFileName();
        try
        {
            string dotted=Path.Combine(Path.GetDirectoryName(path)!,".",Path.GetFileName(path));
            Assert.Equal(path,Path.GetFullPath(dotted));
            Assert.NotEqual(path,dotted);
            Assert.Throws<ArgumentException>(()=>Q35CudaEngine.ResolveNinferGdnChunkPath(dotted));
        }
        finally{File.Delete(path);}
    }
    [Fact]
    public void LoadedModuleAndPrefillPolicyHaveSeparateCacheIdentity()
    {
        string a=new('A',64),b=new('b',64);
        string first=Q35CudaEngine.NinferGdnChunkStateSuffix(CudaNinferGdnChunk.Profile,a,4096);
        Assert.Contains(a.ToLowerInvariant(),first);
        Assert.Contains("ub=4096",first);
        Assert.NotEqual(first,Q35CudaEngine.NinferGdnChunkStateSuffix(CudaNinferGdnChunk.Profile,b,4096));
        Assert.NotEqual(first,Q35CudaEngine.NinferGdnChunkStateSuffix(CudaNinferGdnChunk.Profile,a,2048));
        Assert.NotEqual(first,Q35CudaEngine.NinferGdnChunkStateSuffix("other-valid-profile",a,4096));
        Assert.Equal(first,Q35CudaEngine.NinferGdnChunkStateSuffix(CudaNinferGdnChunk.Profile,a.ToLowerInvariant(),4096));
    }
    [Theory]
    [InlineData("",64,4096)]
    [InlineData("bad|profile",64,4096)]
    [InlineData("bad:profile",64,4096)]
    [InlineData("ok",63,4096)]
    [InlineData("ok",64,63)]
    public void IncompleteIdentityIsRejected(string profile,int shaLength,int rows)
        =>Assert.Throws<ArgumentException>(()=>Q35CudaEngine.NinferGdnChunkStateSuffix(profile,new string('a',shaLength),rows));
    [Fact]
    public void NonHexLoadedIdentityIsRejected()
        =>Assert.Throws<ArgumentException>(()=>Q35CudaEngine.NinferGdnChunkStateSuffix("ok",new string('z',64),4096));
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(63)]
    [InlineData(65)]
    [InlineData(-64)]
    public void StandaloneKernelAdmitsCompleteChunksOnly(int rows)
        =>Assert.Throws<ArgumentOutOfRangeException>(()=>CudaNinferGdnChunk.WorkspaceBytes(rows));
    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(4096)]
    public void WorkspaceRegionsAreAlignedAndDisjoint(int rows)
    {
        var p=CudaNinferGdnChunk.WorkspaceLayout(rows);
        long[] offsets={p.Query,p.Key,p.Value,p.Output,p.CumulativeDecay,p.W,p.U,p.VNew,p.ChunkStates,p.Bytes};
        long[] bytes={(long)rows*16*128*2,(long)rows*16*128*2,(long)rows*48*128*2,(long)rows*48*128*2,
            (long)rows*48*4,(long)rows*48*128*2,(long)rows*48*128*2,(long)rows*48*128*2,(long)(rows/64)*48*128*128*2};
        for(int i=0;i<bytes.Length;i++){Assert.Equal(0,offsets[i]%256);Assert.True(offsets[i]+bytes[i]<=offsets[i+1]);}
        Assert.Equal(0,p.Bytes%256);
    }
}
