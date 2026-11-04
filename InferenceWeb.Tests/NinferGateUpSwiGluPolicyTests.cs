// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class NinferGateUpSwiGluPolicyTests
{
    private static bool Plan(string? option, bool qualified = true, int sms = 170, int context = 40960,
        int rows = 8, bool verify = true, bool graphs = true, bool gemm = true, bool overrides = false, bool matrices = true)
        => Q35CudaEngine.PlanNinferGateUpSwiGlu(true, qualified, sms, context, rows, verify, graphs,
            gemm, overrides, matrices, option!);

    [Fact]
    public void QualifiedRouteHasStrictOptOut()
    {
        Assert.True(Plan(null)); Assert.True(Plan("1")); Assert.False(Plan("0"));
        foreach (string value in new[] { "", "true", " 1", "01", "2" })
            Assert.Throws<ArgumentException>(() => Plan(value));
    }

    [Fact]
    public void OtherDevicesShapesAndProjectionPoliciesKeepTheirOriginalRoute()
    {
        Func<string?, bool>[] profiles = {
            x => Plan(x, qualified: false), x => Plan(x, sms: 169), x => Plan(x, sms: 171),
            x => Plan(x, context: 32768), x => Plan(x, rows: 7), x => Plan(x, verify: false),
            x => Plan(x, graphs: false), x => Plan(x, gemm: false), x => Plan(x, overrides: true),
            x => Plan(x, matrices: false) };
        foreach (var profile in profiles)
        {
            Assert.False(profile(null)); Assert.False(profile("0"));
            Assert.Throws<NotSupportedException>(() => profile("1"));
        }
        Assert.False(Q35CudaEngine.PlanNinferGateUpSwiGlu(false, true, 170, 40960, 8,
            true, true, true, false, true, "invalid"));
    }

    [Fact]
    public void OnlyOriginalGateUpMatricesCanFuse()
    {
        Assert.True(Q35CudaEngine.NinferGateUpSwiGluMatrixEligible(CudaNinferProjectionFormat.NvFp4Planes, true, 8, 34816, 5120));
        foreach (var shape in new[] { (7, 34816, 5120), (9, 34816, 5120), (8, 16384, 5120), (8, 34816, 4096) })
            Assert.False(Q35CudaEngine.NinferGateUpSwiGluMatrixEligible(CudaNinferProjectionFormat.NvFp4Planes, true, shape.Item1, shape.Item2, shape.Item3));
        Assert.False(Q35CudaEngine.NinferGateUpSwiGluMatrixEligible(CudaNinferProjectionFormat.NvFp4Planes, false, 8, 34816, 5120));
        Assert.False(Q35CudaEngine.NinferGateUpSwiGluMatrixEligible(CudaNinferProjectionFormat.BFloat16, true, 8, 34816, 5120));
    }

    [Fact]
    public void GraphsCannotReuseTheUnfusedOrDifferentImageRoute()
    {
        string identity = "|ninfer-gateup-swiglu=" + Q35Kernels.NinferGateUpSwiGluPolicy + ":" + new string('a', 64);
        var old = Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 0, 1024, true);
        var fused = Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 0, 1024, true, gateUpSwiGluIdentity: identity);
        Assert.NotEqual(old, fused);
        Assert.NotEqual(fused, Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 0, 1024, true,
            gateUpSwiGluIdentity: identity.Replace(new string('a', 64), new string('b', 64))));
        foreach (var shape in new[] { (2, 4, true, 0), (1, 7, true, 0), (1, 8, false, 0), (1, 8, true, 1) })
            Assert.Throws<ArgumentException>(() => Q35CudaEngine.ValidateNinferGateUpSwiGluGraphIdentity(shape.Item1, shape.Item2, shape.Item3, shape.Item4, identity));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.ValidateNinferGateUpSwiGluGraphIdentity(1, 8, true, 0, identity.ToUpperInvariant()));
    }
}
