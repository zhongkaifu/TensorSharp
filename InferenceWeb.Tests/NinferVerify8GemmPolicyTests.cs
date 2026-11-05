// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class NinferVerify8GemmPolicyTests
{
    [Theory]
    [InlineData(null, true)] [InlineData("1", true)] [InlineData("0", false)]
    public void QualifiedVerificationHasStrictOptOut(string? value, bool enabled)
        => Assert.Equal(enabled, Q35CudaEngine.PlanNinferVerify8Gemm(true, true, 170, true, 8, value!));

    [Theory]
    [InlineData("")] [InlineData("true")] [InlineData(" 1")] [InlineData("01")] [InlineData("2")]
    public void MalformedNinferControlsDoNotSilentlyFallBack(string option)
        => Assert.Throws<ArgumentException>(() => Q35CudaEngine.PlanNinferVerify8Gemm(true, false, 0, false, 0, option));

    [Theory]
    [InlineData(false, 170, true, 8)] [InlineData(true, 169, true, 8)]
    [InlineData(true, 171, true, 8)] [InlineData(true, 170, false, 8)] [InlineData(true, 170, true, 7)]
    public void UnsupportedProfilesKeepOldDefaultAndExplicitOneRefuses(bool qualified, int sms, bool verify, int rows)
    {
        Assert.False(Q35CudaEngine.PlanNinferVerify8Gemm(true, qualified, sms, verify, rows, null!));
        Assert.False(Q35CudaEngine.PlanNinferVerify8Gemm(true, qualified, sms, verify, rows, "0"));
        Assert.Throws<NotSupportedException>(() => Q35CudaEngine.PlanNinferVerify8Gemm(true, qualified, sms, verify, rows, "1"));
    }

    [Theory]
    [InlineData(null)] [InlineData("0")] [InlineData("1")] [InlineData("invalid")]
    public void NewControlDoesNotApplyToGguf(string? option)
        => Assert.False(Q35CudaEngine.PlanNinferVerify8Gemm(false, true, 170, true, 8, option!));

    [Theory]
    [InlineData(16384, 5120)] [InlineData(34816, 5120)] [InlineData(14336, 5120)]
    [InlineData(5120, 17408)] [InlineData(5120, 6144)]
    public void CompleteT8UsesOnlyTheEnumeratedProjectionShapes(int n, int k)
    {
        Assert.True(Q35Kernels.UseNinferVerify8Gemm(true, true, false, 8, n, k));
        Assert.False(Q35Kernels.UseNinferVerify8Gemm(false, true, false, 8, n, k));
        Assert.False(Q35Kernels.UseNinferVerify8Gemm(true, false, false, 8, n, k));
        Assert.False(Q35Kernels.UseNinferVerify8Gemm(true, true, true, 8, n, k));
        Assert.False(Q35Kernels.UseNinferVerify8Gemm(true, true, false, 7, n, k));
        Assert.False(Q35Kernels.UseNinferVerify8Gemm(true, true, false, 9, n, k));
    }

    [Theory]
    [InlineData(248320, 5120)] [InlineData(96, 5120)] [InlineData(16384, 4096)]
    public void HeadGateAndOtherModelShapesKeepOriginalKernel(int n, int k)
        => Assert.False(Q35Kernels.UseNinferVerify8Gemm(true, true, false, 8, n, k));

    [Theory]
    [InlineData(false, false)] [InlineData(true, false)] [InlineData(false, true)] [InlineData(true, true)]
    public void PolicyCannotChangeAfterGemmOrPriorConfiguration(bool configured, bool submitted)
    {
        if (configured || submitted)
            Assert.Throws<InvalidOperationException>(() => Q35Kernels.ValidateNinferVerify8GemmConfiguration(configured, submitted));
        else Q35Kernels.ValidateNinferVerify8GemmConfiguration(configured, submitted);
    }
}
