// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class NinferVerify8AttentionEnginePolicyTests
{
    private sealed class ModuleFile : IDisposable
    {
        private readonly string _directory = Directory.CreateTempSubdirectory("ninfer-verify8-policy-").FullName;
        internal string Path { get; }
        internal ModuleFile() { Path = System.IO.Path.Combine(_directory, "explicit.fatbin"); File.WriteAllBytes(Path, new byte[] { 1 }); }
        public void Dispose() => Directory.Delete(_directory, true);
    }

    [Fact]
    public void AbsentFlagStaysOffWithoutReadingFilesOrAdmittingGeometry()
        => Assert.Null(CudaNinferAttention.ResolveVerify8Path(null!, false, 0, 0, 0, 0, 0, 0));

    [Fact]
    public void ExplicitPathAdmitsNormal65536ContextWithoutChangingVisibleCap()
    {
        using var f = new ModuleFile();
        Assert.Equal(f.Path, CudaNinferAttention.ResolveVerify8Path(f.Path, true, 2048, 65536, 170, 24, 4, 256));
        Assert.Equal(f.Path, CudaNinferAttention.ResolveVerify8Path(f.Path, true, 8, 262144, 170, 24, 4, 256));
        Assert.Equal(640, CudaNinferPrefillAttentionModule.PlanVerify8(202, 257).Pages);
        Assert.False(CudaNinferAttention.UseVerify8(true, 8, 1, true, true, 24, 4, 40961));
    }

    [Theory]
    [InlineData("")]
    [InlineData(" ")]
    [InlineData("0")]
    [InlineData("1")]
    [InlineData("relative.fatbin")]
    [InlineData("/tmp/parent/../candidate.fatbin")]
    public void NoncanonicalAndBooleanLikePathsNeverEnableOrSilentlyFallBack(string raw)
        => Assert.Throws<ArgumentException>(() => CudaNinferAttention.ResolveVerify8Path(raw, true, 2048, 65536, 170, 24, 4, 256));

    [Theory]
    [InlineData(false, 2048, 65536, 170, 24, 4, 256)]
    [InlineData(true, 7, 65536, 170, 24, 4, 256)]
    [InlineData(true, 2048, 7, 170, 24, 4, 256)]
    [InlineData(true, 2048, 262145, 170, 24, 4, 256)]
    [InlineData(true, 2048, 65536, 169, 24, 4, 256)]
    [InlineData(true, 2048, 65536, 171, 24, 4, 256)]
    [InlineData(true, 2048, 65536, 170, 16, 4, 256)]
    [InlineData(true, 2048, 65536, 170, 24, 2, 256)]
    [InlineData(true, 2048, 65536, 170, 24, 4, 128)]
    public void UnsupportedExplicitGeometryFailsBeforeCudaLoading(bool verify, int rows, int context,
        int sms, int heads, int kvHeads, int headDim)
    {
        using var f = new ModuleFile();
        Assert.Throws<NotSupportedException>(() => CudaNinferAttention.ResolveVerify8Path(f.Path, verify,
            rows, context, sms, heads, kvHeads, headDim));
    }

    [Fact]
    public void MissingExplicitAssetFailsWithoutAnAutomaticFallback()
    {
        using var f = new ModuleFile(); File.Delete(f.Path);
        Assert.Throws<FileNotFoundException>(() => CudaNinferAttention.ResolveVerify8Path(f.Path, true, 2048, 65536, 170, 24, 4, 256));
    }

    [Theory]
    [InlineData(true, 8, 1, true, true, 24, 4, 8, true)]
    [InlineData(true, 8, 1, true, true, 24, 4, 40960, true)]
    [InlineData(false, 8, 1, true, true, 24, 4, 202, false)]
    [InlineData(true, 7, 1, true, true, 24, 4, 202, false)]
    [InlineData(true, 9, 1, true, true, 24, 4, 202, false)]
    [InlineData(true, 8, 2, true, true, 24, 4, 202, false)]
    [InlineData(true, 8, 1, false, true, 24, 4, 202, false)]
    [InlineData(true, 8, 1, true, false, 24, 4, 202, false)]
    [InlineData(true, 8, 1, true, true, 16, 4, 202, false)]
    [InlineData(true, 8, 1, true, true, 24, 2, 202, false)]
    [InlineData(true, 8, 1, true, true, 24, 4, 7, false)]
    [InlineData(true, 8, 1, true, true, 24, 4, 40961, false)]
    public void DispatchAndGraphPolicyUseTheSameNarrowPredicate(bool enabled, int rows, int sequences,
        bool verify, bool valid, int heads, int kvHeads, int visible, bool expected)
        => Assert.Equal(expected, CudaNinferAttention.UseVerify8(enabled, rows, sequences, verify, valid, heads, kvHeads, visible));

    [Fact]
    public void Verify8GraphKeyIsExplicitAndStableAcrossLegacySplitBoundaries()
    {
        var first = Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 0, 74, true);
        Assert.True(first.Verify8Attention); Assert.Equal(0, first.AttentionChunk); Assert.Equal(4, first.AttentionSplits);
        foreach (int visible in new[] { 74, 128, 129, 202, 256, 257, 512, 1024, 40960 })
        {
            Assert.Equal(first, Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 0, visible, true));
            var old = Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 0, visible, false);
            var expectedOld = Q35Kernels.GetSplitAttentionPlan(visible);
            Assert.False(old.Verify8Attention); Assert.Equal(expectedOld.Chunk, old.AttentionChunk); Assert.Equal(expectedOld.Splits, old.AttentionSplits);
            Assert.NotEqual(first, old);
        }
        Assert.NotEqual(first, first with { Verify8Attention = false });
        Assert.NotEqual(first, Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 1, 74, true));
        Assert.False(Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 0, 40961, false).Verify8Attention);
    }

    [Theory]
    [InlineData(2, 8, true, 202)]
    [InlineData(1, 7, true, 202)]
    [InlineData(1, 8, false, 202)]
    [InlineData(1, 8, true, 40961)]
    public void InconsistentCapturedRouteMetadataFailsClosed(int lanes, int window, bool verify, int keys)
        => Assert.Throws<ArgumentException>(() => Q35CudaEngine.PlanNinferDecodeGraphKey(lanes, window, verify, 0, keys, true));
}
