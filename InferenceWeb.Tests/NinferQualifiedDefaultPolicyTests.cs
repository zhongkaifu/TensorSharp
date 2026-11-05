// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;

namespace InferenceWeb.Tests;

/// <summary>Automatic selection/admission only. These tests do not initialize a
/// CUDA device or qualify the new default runtime composition on a model.</summary>
public sealed class NinferQualifiedDefaultPolicyTests
{
    [Theory]
    [InlineData(null, true)] [InlineData("1", true)] [InlineData("0", false)]
    public void AutomaticControlsDefaultOnWithAnExactOptOut(string? value, bool expected)
        => Assert.Equal(expected, Q35CudaEngine.NinferAutomaticOptimizationRequested(value, "test-option"));

    [Theory]
    [InlineData("")] [InlineData(" ")] [InlineData("true")] [InlineData("2")]
    [InlineData("01")] [InlineData(" 1")] [InlineData("1 ")] [InlineData("１")]
    public void MalformedAutomaticControlsCannotSilentlySelectAnImplementation(string value)
        => Assert.Throws<ArgumentException>(() => Q35CudaEngine.NinferAutomaticOptimizationRequested(value, "test-option"));

    [Theory]
    [InlineData(12, 0, true)] [InlineData(12, 1, false)] [InlineData(12, 2, false)]
    [InlineData(10, 0, false)] [InlineData(9, 0, false)] [InlineData(8, 0, false)]
    [InlineData(13, 0, false)] [InlineData(0, 0, false)]
    public void OnlyTheValidatedDeviceGenerationReceivesAutomaticDefaults(int major, int minor, bool expected)
        => Assert.Equal(expected, Geometry(true, 64, 5120, 17408, 248320, 24, 4, 256, major, minor));

    [Theory]
    [InlineData(false, 64, 5120, 17408, 248320, 24, 4, 256)]
    [InlineData(true, 63, 5120, 17408, 248320, 24, 4, 256)]
    [InlineData(true, 64, 4096, 17408, 248320, 24, 4, 256)]
    [InlineData(true, 64, 5120, 17407, 248320, 24, 4, 256)]
    [InlineData(true, 64, 5120, 17408, 248077, 24, 4, 256)]
    [InlineData(true, 64, 5120, 17408, 248320, 16, 4, 256)]
    [InlineData(true, 64, 5120, 17408, 248320, 24, 2, 256)]
    [InlineData(true, 64, 5120, 17408, 248320, 24, 4, 128)]
    public void AutomaticDefaultsDoNotLeakIntoGgufOrOtherModelRecipes(bool ninfer, int layers,
        int hidden, int ff, int vocabulary, int heads, int kvHeads, int headDim)
        => Assert.False(Geometry(ninfer, layers, hidden, ff, vocabulary, heads, kvHeads, headDim, 12, 0));

    private static bool Geometry(bool ninfer, int layers, int hidden, int ff, int vocabulary,
        int heads, int kvHeads, int headDim, int major, int minor)
        => Q35CudaEngine.NinferQualifiedDefaultGeometry(ninfer, layers, hidden, ff,
            vocabulary, heads, kvHeads, headDim, major, minor);

    [Theory]
    [InlineData(true)] [InlineData(false)]
    public void IndependentOptOutsAndExplicitGraphPolicyRemainIndependent(bool qualified)
    {
        var defaults = Q35CudaEngine.PlanNinferQualifiedDefaults(qualified, null, null, null);
        Assert.Equal(qualified, defaults.AutoA16); Assert.Equal(qualified, defaults.AutoRepack);
        Assert.Equal(qualified, defaults.DecodeGraphs);
        var vectorOff = Q35CudaEngine.PlanNinferQualifiedDefaults(qualified, "0", null, null);
        Assert.False(vectorOff.AutoA16); Assert.Equal(qualified, vectorOff.AutoRepack);
        Assert.Equal(qualified, vectorOff.DecodeGraphs);
        var repackOff = Q35CudaEngine.PlanNinferQualifiedDefaults(qualified, null, "0", null);
        Assert.Equal(qualified, repackOff.AutoA16); Assert.False(repackOff.AutoRepack);
        var graphOff = Q35CudaEngine.PlanNinferQualifiedDefaults(qualified, null, null, "0");
        Assert.False(graphOff.DecodeGraphs); Assert.Equal("disabled", graphOff.GraphSelection);
        var graphOn = Q35CudaEngine.PlanNinferQualifiedDefaults(qualified, null, null, "1");
        Assert.True(graphOn.DecodeGraphs); Assert.Equal("explicit", graphOn.GraphSelection);
        var explicitAuto = Q35CudaEngine.PlanNinferQualifiedDefaults(qualified, "1", "1", null);
        Assert.Equal(qualified, explicitAuto.AutoA16); Assert.Equal(qualified, explicitAuto.AutoRepack);
    }

    [Theory]
    [InlineData("/original.fatbin", null)] [InlineData(null, "/legacy.fatbin")]
    [InlineData("", null)] [InlineData(null, " ")]
    public void AnExplicitA16PolicyAlwaysSuppressesAutomaticVectorDiscovery(string? original, string? legacy)
    {
        Assert.False(Q35CudaEngine.UseAutomaticNinferA16(true, original, legacy));
        Assert.False(Q35CudaEngine.UseAutomaticNinferA16(false, original, legacy));
    }

    [Fact]
    public void NoOptOutCanHideTwoConflictingExplicitA16Policies()
    {
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.UseAutomaticNinferA16(true, "/original", "/legacy"));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.UseAutomaticNinferA16(false, "/original", "/legacy"));
        Assert.True(Q35CudaEngine.UseAutomaticNinferA16(true, null, null));
        Assert.False(Q35CudaEngine.UseAutomaticNinferA16(false, null, null));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.NormalizeNinferA16ModulePath("0"));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.ResolveNinferLoadRepackPath("0", true));
    }

    [Fact]
    public void StandardAssetDiscoveryPreservesPairedPtxLoadingAndSupportsFatbinOnlyPackages()
    {
        const string name = "tensorsharp_test.ptx";
        var calls = new List<string>();
        string WithPtx(string candidate) { calls.Add(candidate); return candidate.EndsWith(".ptx") ? "/assets/cuda_kernels/tensorsharp_test.ptx" : "/assets/cuda_kernels/tensorsharp_test.fatbin"; }
        Assert.Equal("/assets/cuda_kernels/tensorsharp_test.ptx", Q35CudaEngine.LocateNinferAutomaticAsset(name, WithPtx));
        Assert.Equal(new[] { name }, calls);
        calls.Clear();
        string FatbinOnly(string candidate) { calls.Add(candidate); return candidate.EndsWith(".fatbin") ? "/assets/cuda_kernels/tensorsharp_test.fatbin" : null; }
        Assert.Equal("/assets/cuda_kernels/tensorsharp_test.fatbin", Q35CudaEngine.LocateNinferAutomaticAsset(name, FatbinOnly));
        Assert.Equal(new[] { name, "tensorsharp_test.fatbin" }, calls);
        Assert.Null(Q35CudaEngine.LocateNinferAutomaticAsset(name, _ => null));
    }

    [Theory]
    [InlineData(200)] [InlineData(209)] [InlineData(218)] [InlineData(222)] [InlineData(500)]
    public void IncompatibleAutomaticModuleImagesCanFallBackBeforeUse(int error)
        => Assert.True(Q35CudaEngine.NinferAutomaticModuleUnavailable(new CudaException(error, "admission")));

    [Theory]
    [InlineData(2)] [InlineData(201)] [InlineData(700)] [InlineData(702)] [InlineData(719)] [InlineData(999)]
    public void ResourceContextAndExecutionFaultsCannotBeHiddenAsMissingAutomaticAssets(int error)
        => Assert.False(Q35CudaEngine.NinferAutomaticModuleUnavailable(new CudaException(error, "fatal")));

    [Fact]
    public void OnlyKnownAdmissionFailuresMayFallback()
    {
        Assert.True(Q35CudaEngine.NinferAutomaticModuleUnavailable(new FileNotFoundException()));
        Assert.True(Q35CudaEngine.NinferAutomaticModuleUnavailable(new UnauthorizedAccessException()));
        Assert.True(Q35CudaEngine.NinferAutomaticModuleUnavailable(new InvalidOperationException("wrong module ABI")));
        Assert.False(Q35CudaEngine.NinferAutomaticModuleUnavailable(new OutOfMemoryException()));
        Assert.False(Q35CudaEngine.NinferAutomaticModuleUnavailable(new AccessViolationException()));
        Assert.False(Q35CudaEngine.NinferAutomaticModuleUnavailable(new ArgumentException()));
    }
}
