// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class NinferVerify8AutomaticPolicyTests
{
    [Theory]
    [InlineData(null, true)] [InlineData("1", true)] [InlineData("0", false)]
    public void QualifiedVerify8HasStrictOptOut(string? option, bool expected)
        => Assert.Equal(expected, Q35CudaEngine.PlanNinferAutomaticVerify8(true, 170, true, 2048, 65536, option!));

    [Theory]
    [InlineData("")] [InlineData("true")] [InlineData("2")] [InlineData(" 1")] [InlineData("01")]
    public void MalformedControlsFailEvenWhenHardwareOrVerificationWouldBeIneligible(string option)
        => Assert.Throws<ArgumentException>(() => Q35CudaEngine.PlanNinferAutomaticVerify8(false, 0, false, 0, 0, option));

    [Theory]
    [InlineData(false, 170, true, 2048, 65536)]
    [InlineData(true, 169, true, 2048, 65536)]
    [InlineData(true, 171, true, 2048, 65536)]
    [InlineData(true, 170, false, 2048, 65536)]
    [InlineData(true, 170, true, 7, 65536)]
    [InlineData(true, 170, true, 8, 7)]
    [InlineData(true, 170, true, 2048, 262145)]
    public void ExplicitAutomaticOneNeverExpandsCoverageOrEnablesVerification(bool qualified, int sms,
        bool verification, int rows, int context)
        => Assert.False(Q35CudaEngine.PlanNinferAutomaticVerify8(qualified, sms, verification, rows, context, "1"));

    [Theory]
    [InlineData(12, 0, 64, true)] [InlineData(12, 1, 64, false)] [InlineData(12, 0, 63, false)]
    public void AutomaticGateIncludesFullExistingRecipeAndDeviceGeneration(int major, int minor, int layers, bool expected)
    {
        bool qualified = Q35CudaEngine.NinferQualifiedDefaultGeometry(true, layers, 5120, 17408, 248320, 24, 4, 256, major, minor);
        Assert.Equal(expected, Q35CudaEngine.PlanNinferAutomaticVerify8(qualified, 170, true, 2048, 40960, null!));
    }

    [Theory]
    [InlineData(null)] [InlineData("0")] [InlineData("1")]
    public void AnExplicitModuleSuppressesAutomaticDiscoveryEvenWithAutomaticZero(string? option)
    {
        bool automatic = Q35CudaEngine.PlanNinferAutomaticVerify8(true, 170, true, 2048, 40960, option!);
        Assert.False(CudaNinferAttention.UseAutomaticVerify8(automatic, "/explicit/module.fatbin"));
        // Malformed explicit paths are still owned by the explicit parser.
        Assert.False(CudaNinferAttention.UseAutomaticVerify8(automatic, "0"));
        Assert.Throws<ArgumentException>(() => CudaNinferAttention.ResolveVerify8Path("0", true, 2048, 40960, 170, 24, 4, 256));
        Assert.Equal(option != "0", CudaNinferAttention.UseAutomaticVerify8(automatic, null!));
    }

    [Fact]
    public void PackagedModuleUsesExistingPairedPtxResolverAndFatbinOnlyFallback()
    {
        string name = Q35CudaEngine.NinferQualifiedVerify8Asset;
        var requests = new List<string>();
        string? Resolve(string file) { requests.Add(file); return file.EndsWith(".fatbin") ? "/assets/cuda_kernels/" + file : null; }
        Assert.EndsWith("/tensorsharp_ninfer_prefill_attention.fatbin", Q35CudaEngine.LocateNinferAutomaticAsset(name, Resolve!));
        Assert.Equal(new[] { name, "tensorsharp_ninfer_prefill_attention.fatbin" }, requests);
        Assert.Null(Q35CudaEngine.LocateNinferAutomaticAsset(name, _ => null!));
    }
}
