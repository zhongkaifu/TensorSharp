// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

/// <summary>CPU admission/scratch contracts. Passing these does not qualify a CUDA component.</summary>
public sealed class NinferDraftModuleContractTests
{
    [Theory]
    [InlineData(0, true, 1)]
    [InlineData(96, true, 1)]
    [InlineData(97, false, 4)]
    [InlineData(1024, false, 32)]
    [InlineData(2047, false, 32)]
    [InlineData(2048, false, 32)]
    [InlineData(250000, false, 32)]
    public void OriginalSwaEnvelopeControlsDirectAndSplitScratch(int context, bool direct, int splits)
    {
        var p = CudaNinferDraftModule.ValidateSwa(context, 8, 3, 16);
        Assert.Equal(direct, p.Direct); Assert.Equal(splits, p.Splits);
        Assert.Equal(196608, p.QueryBytes); Assert.Equal(49152, p.KeyValueBytes); Assert.Equal(67108864, p.RingBytes);
        Assert.Equal(direct ? 0 : p.QueryBytes * splits, p.PartialBytes);
        Assert.Equal(direct ? 0 : 3072 * splits, p.StatBytes);
    }

    [Theory]
    [InlineData(-1, 8, 1, 1)]
    [InlineData(1024, 0, 1, 1)]
    [InlineData(1024, 9, 1, 1)]
    [InlineData(1024, 8, 9, 16)]
    [InlineData(1024, 8, 4, 3)]
    [InlineData(1024, 8, 4, 17)]
    [InlineData(int.MaxValue, 8, 1, 1)]
    public void SwaBoundsFailBeforeAllocatingScratch(int context, int block, int batch, int slots)
        => Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferDraftModule.ValidateSwa(context, block, batch, slots));

    [Fact]
    public void FullVocabularyTop16UsesBoundedTwoPassOriginalSelectionWorkspace()
    {
        Assert.Equal(15616, CudaNinferDraftModule.Top16WorkspaceBytes(248320, 1));
        Assert.Equal(874496, CudaNinferDraftModule.Top16WorkspaceBytes(248320, 56));
        Assert.Equal(128, CudaNinferDraftModule.Top16WorkspaceBytes(16, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferDraftModule.Top16WorkspaceBytes(15, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferDraftModule.Top16WorkspaceBytes(248321, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferDraftModule.Top16WorkspaceBytes(248320, 57));
    }

    [Fact]
    public void DrafterFull128RotaryFrequenciesUseUnscaledDoubleBase1e7()
    {
        double[] f = CudaNinferDraftModule.RopeFrequencies(); Assert.Equal(64, f.Length); Assert.Equal(1.0, f[0]);
        Assert.InRange(Math.Abs(1.0 / Math.Sqrt(1e7) - f[32]), 0, 1e-18);
        for (int i = 1; i < f.Length; i++) Assert.True(double.IsFinite(f[i]) && f[i] > 0 && f[i] < f[i - 1]);
        f[0] = 2; Assert.Equal(1.0, CudaNinferDraftModule.RopeFrequencies()[0]);
    }
}
