// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

// This CUDA policy's validation build must enable TENSORSHARP_GGML_NATIVE_BUILD_TESTS.
// A missing native observation hook fails the prerequisite, never counts as a pass.
public sealed class QwenQ8PrecisionLifetimeTests
{
    [GgmlFact(BackendType.GgmlCuda)]
    public void HostAndDeviceKeys_RetireIndependentlyAndPreserveOtherOwners()
    {
        using var weight = new QuantizedWeight(new byte[34], (int)GgmlTensorType.Q8_0, 32, 1);
        weight.EnableQ8F32Activations();
        using var native = new RegistrationCounts();
        IntPtr host = weight.Data;
        Assert.Equal(1, native.Read(host)); // Data and CacheKey initially alias.
        weight.EnableQ8F32Activations();
        Assert.Equal(1, native.Read(host));
        GgmlQ8Precision.RegisterWeight(host);
        try
        {
            Assert.Equal(2, native.Read(host));
            IntPtr device = weight.EnsureDeviceCacheKey();
            Assert.NotEqual(host, device);
            Assert.Equal(1, native.Read(device));
            weight.ReleaseHostData();
            Assert.Equal(1, native.Read(host)); // The independent registry owner remains.
            Assert.Equal(1, native.Read(device));
            weight.Dispose();
            Assert.Equal(0, native.Read(device));
            Assert.Equal(1, native.Read(host));
            weight.Dispose();
        }
        finally { GgmlQ8Precision.UnregisterWeight(host); }
        Assert.Equal(0, native.Read(host));
    }

    [GgmlFact(BackendType.GgmlCuda)]
    public void DeclinedDevicePreload_RetiresOpaqueKeyAndKeepsHostPolicy()
    {
        using var weight = new QuantizedWeight(new byte[34], (int)GgmlTensorType.Q8_0, 32, 1);
        weight.EnableQ8F32Activations();
        using var native = new RegistrationCounts();
        IntPtr host = weight.Data;
        IntPtr device = weight.EnsureDeviceCacheKey();
        weight.MarkDevicePreloadTooLarge();
        Assert.Equal(host, weight.CacheKey);
        Assert.Equal(0, native.Read(device));
        Assert.Equal(1, native.Read(host));
        weight.DisableQ8F32Activations();
        Assert.Equal(0, native.Read(host));
        weight.EnableQ8F32Activations();
        Assert.Equal(1, native.Read(host));
        weight.Dispose();
        Assert.Equal(0, native.Read(host));
    }

    [GgmlFact(BackendType.GgmlCuda)]
    public void FailedDeviceKeyRegistration_RollsBackIdentityAndCanRetry()
    {
        using var weight = new QuantizedWeight(new byte[34], (int)GgmlTensorType.Q8_0, 32, 1);
        weight.EnableQ8F32Activations();
        using var native = new RegistrationCounts();
        IntPtr host = weight.Data;
        native.FailNext();
        Assert.Throws<InvalidOperationException>(() => weight.EnsureDeviceCacheKey());
        Assert.Equal(host, weight.CacheKey);
        Assert.Equal(1, native.Read(host));
        IntPtr device = weight.EnsureDeviceCacheKey();
        Assert.NotEqual(host, device);
        Assert.Equal(1, native.Read(device));
        weight.Dispose();
        Assert.Equal(0, native.Read(device));
        Assert.Equal(0, native.Read(host));
    }

    private sealed class RegistrationCounts : IDisposable
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate long Count(IntPtr key);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void FailRegistration();
        private readonly IntPtr _module;
        private readonly Count _count;
        public RegistrationCounts()
        {
            _module = NativeLibrary.Load(Qwen4ExpExpertCacheScenario.MappedNativePath());
            try
            {
                _count = Marshal.GetDelegateForFunctionPointer<Count>(NativeLibrary.GetExport(
                    _module, "TSGgml_TestQ8F32WeightRegistrationCount"));
            }
            catch { NativeLibrary.Free(_module); throw; }
        }
        public long Read(IntPtr key) => _count(key);
        public void FailNext() => Marshal.GetDelegateForFunctionPointer<FailRegistration>(NativeLibrary.GetExport(
            _module, "TSGgml_TestQ8F32FailNextRegistration"))();
        public void Dispose() => NativeLibrary.Free(_module);
    }
}
