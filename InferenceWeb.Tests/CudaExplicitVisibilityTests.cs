// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class CudaExplicitVisibilityTests
{
    [Theory]
    [InlineData(null, false)]
    [InlineData("", true)]
    [InlineData(" ", true)]
    [InlineData("-1", true)]
    [InlineData(" \t-1 ", true)]
    [InlineData("0", false)]
    [InlineData("0,-1", false)]
    [InlineData("GPU-01234567", false)]
    [InlineData("MIG-01234567", false)]
    public void OnlyAllHiddenMasksBypassTheDriver(string? mask, bool hidden)
        => Assert.Equal(hidden, CudaDevice.VisibilityMaskHidesAllDevices(mask));

    [Theory]
    [InlineData("-1")]
    [InlineData(" \t-1 ")]
    [InlineData(" ")]
    public void ExplicitlyHiddenDevicesAreUnavailableWithoutInitialization(string mask)
    {
        using var environment = new NativeEnvScope();
        environment.Set("CUDA_VISIBLE_DEVICES", mask);
        Assert.False(CudaDevice.IsAvailable());
        Assert.Equal(0, CudaDevice.GetDeviceCount());
        var error = Assert.Throws<InvalidOperationException>(() => CudaDevice.GetDevice(0));
        Assert.Contains("CUDA_VISIBLE_DEVICES", error.Message);
    }
}
