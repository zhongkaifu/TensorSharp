// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using TensorSharp.Cuda;
using Xunit;

public sealed class Qwen35NinferExactVerifyPolicyTests
{
    [Fact]
    public void UnsetOptionDoesNotDiscoverCandidate()
        => Assert.Null(Q35CudaEngine.ResolveNinferExactVerifyPath(null));

    [Theory]
    [InlineData("")]
    [InlineData(" ")]
    [InlineData("\t")]
    [InlineData("candidate.fatbin")]
    public void ExplicitOptionRequiresNormalizedAbsolutePath(string raw)
        => Assert.Throws<ArgumentException>(() => Q35CudaEngine.ResolveNinferExactVerifyPath(raw));

    [Fact]
    public void MissingImageFailsWithoutFallback()
        => Assert.Throws<FileNotFoundException>(() => Q35CudaEngine.ResolveNinferExactVerifyPath(
            Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".fatbin")));

    [Fact]
    public void OnlyExistingNormalizedPathIsAccepted()
    {
        string path = Path.GetTempFileName();
        try
        {
            path = Path.GetFullPath(path);
            Assert.Equal(path, Q35CudaEngine.ResolveNinferExactVerifyPath(path));
            string relative = Path.GetRelativePath(Environment.CurrentDirectory, path);
            Assert.Throws<ArgumentException>(() => Q35CudaEngine.ResolveNinferExactVerifyPath(relative));
            string dotted = Path.Combine(Path.GetDirectoryName(path)!, ".", Path.GetFileName(path));
            Assert.NotEqual(path, dotted);
            Assert.Throws<ArgumentException>(() => Q35CudaEngine.ResolveNinferExactVerifyPath(dotted));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void StateIdentityPinsPolicyWholeImageAndPhysicalLimit()
    {
        string upper = new('A', 64), other = new('b', 64);
        string suffix = Q35CudaEngine.BuildNinferExactVerifyStateSuffix(upper);
        Assert.Contains(Q35CudaEngine.NinferExactVerifyArithmeticPolicy, suffix);
        Assert.Contains(upper.ToLowerInvariant(), suffix);
        Assert.EndsWith(":maxrows=8", suffix);
        Assert.Equal(suffix, Q35CudaEngine.BuildNinferExactVerifyStateSuffix(upper.ToLowerInvariant()));
        Assert.NotEqual(suffix, Q35CudaEngine.BuildNinferExactVerifyStateSuffix(other));
        Assert.NotEqual(suffix, Q35CudaEngine.BuildNinferA16StateSuffix(upper));
    }

    [Theory]
    [InlineData("")]
    [InlineData("abcd")]
    [InlineData("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaz")]
    public void IncompleteOrNonHexIdentityFails(string sha)
        => Assert.Throws<ArgumentException>(() => Q35CudaEngine.BuildNinferExactVerifyStateSuffix(sha));

    [Theory]
    [InlineData(1, 1)]
    [InlineData(1, 8)]
    [InlineData(2, 4)]
    [InlineData(3, 2)]
    [InlineData(4, 2)]
    [InlineData(8, 1)]
    public void LimitCountsAllPhysicalLanesAndColumns(int lanes, int window)
        => Q35CudaEngine.ValidateNinferExactVerifyGeometry(true, lanes, window);

    [Theory]
    [InlineData(1, 9)]
    [InlineData(2, 5)]
    [InlineData(3, 3)]
    [InlineData(4, 3)]
    [InlineData(9, 1)]
    [InlineData(0, 1)]
    [InlineData(1, 0)]
    [InlineData(-1, 1)]
    [InlineData(1, -1)]
    [InlineData(int.MaxValue, int.MaxValue)]
    public void EnabledGeometryRejectsBeforeTargetMutation(int lanes, int window)
        => Assert.Throws<ArgumentOutOfRangeException>(() => Q35CudaEngine.ValidateNinferExactVerifyGeometry(true, lanes, window));

    [Fact]
    public void DisabledGeometryDoesNotConstrainExistingPolicy()
        => Q35CudaEngine.ValidateNinferExactVerifyGeometry(false, 4, 16);

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    public void VerificationNv4UsesPerTokenLegacyScheduleAtEverySupportedBatch(int rows)
    {
        foreach (int columns in new[] { 5120, 6144, 17408 })
            foreach (int threshold in new[] { 2, 4 })
                Assert.True(Q35CudaEngine.ShouldUseNinferExactVerify(true, true,
                    CudaNinferProjectionFormat.NvFp4Planes, 128, columns, rows, threshold));
    }

    [Fact]
    public void OrdinaryAndDisabledPathsRemainUnconstrained()
    {
        Assert.False(Q35CudaEngine.ShouldUseNinferExactVerify(true, false,
            CudaNinferProjectionFormat.NvFp4Planes, 128, 5120, 4096, 4));
        Assert.False(Q35CudaEngine.ShouldUseNinferExactVerify(false, true,
            CudaNinferProjectionFormat.NvFp4Planes, 128, 5120, 64, 4));
    }

    [Fact]
    public void GdnInputAndBf16ProjectionKeepExistingArithmetic()
    {
        Assert.False(Q35CudaEngine.ShouldUseNinferExactVerify(true, true,
            CudaNinferProjectionFormat.NvFp4Planes, 16384, 5120, 8, 1));
        Assert.False(Q35CudaEngine.ShouldUseNinferExactVerify(true, true,
            CudaNinferProjectionFormat.BFloat16, 96, 5120, 8, 4));
    }

    [Theory]
    [InlineData(CudaNinferProjectionFormat.NvFp4Planes, 1)]
    [InlineData(CudaNinferProjectionFormat.NvFp4Planes, 4)]
    [InlineData(CudaNinferProjectionFormat.BFloat16, 4)]
    public void MatrixExclusionCannotBypassEnabledPhysicalLimit(CudaNinferProjectionFormat format, int threshold)
        => Assert.Throws<ArgumentOutOfRangeException>(() => Q35CudaEngine.ShouldUseNinferExactVerify(true, true,
            format, 128, 5120, 9, threshold));

    [Theory]
    [InlineData(128, 4096)]
    [InlineData(0, 5120)]
    public void EligibleUnsupportedProjectionFailsClosed(int outputRows, int columns)
        => Assert.Throws<NotSupportedException>(() => Q35CudaEngine.ShouldUseNinferExactVerify(true, true,
            CudaNinferProjectionFormat.NvFp4Planes, outputRows, columns, 8, 4));
}
