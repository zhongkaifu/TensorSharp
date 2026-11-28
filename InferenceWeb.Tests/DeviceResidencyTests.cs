// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.Graph;

namespace InferenceWeb.Tests;

/// <summary>
/// D02: the owner of host bytes a backend keeps device copies of releases them through DeviceResidency, which knows which
/// backend keeps which copy (a resident weight cache by cache key, an executor's pointer cache by address): the copies go
/// before the bytes, and an address the bytes are reused at binds the new values.
/// </summary>
public sealed class DeviceResidencyTests
{
    /// <summary>A backend with a resident weight cache (as CUDA and MLX register theirs).</summary>
    private sealed class CachingAllocator : IAllocator
    {
        public BlasEnum BlasEnum => BlasEnum.DotNet;
        public int DeviceId => 0;
        public Storage Allocate(DType elementType, long elementCount) => throw new NotSupportedException();
        public float GetAllocatedMemoryRatio() => 0;
    }

    /// <summary>A backend whose executor binds host pointers (as the GGML executors do).</summary>
    private sealed class BindingAllocator : IAllocator
    {
        public BlasEnum BlasEnum => BlasEnum.DotNet;
        public int DeviceId => 0;
        public Storage Allocate(DType elementType, long elementCount) => throw new NotSupportedException();
        public float GetAllocatedMemoryRatio() => 0;
    }

    private sealed class RecordingExecutor : IGraphExecutor
    {
        public RecordingExecutor(bool binds, List<string> log) { Capabilities = new GraphCapabilities { BindsHostPointers = binds }; Log = log; }
        public List<string> Log { get; }
        public GraphCapabilities Capabilities { get; }
        public IGraphPlan Compile(GraphProgram program) => throw new NotSupportedException();
        public void SyncToHost(IntPtr data, long bytes) { }
        public void Invalidate(IntPtr data) => Log.Add($"invalidate {data}");
    }

    private static readonly List<string> s_log = new();

    static DeviceResidencyTests()
    {
        DeviceResidency.RegisterWeightReleaser(typeof(CachingAllocator), (allocator, key) => s_log.Add($"release {key}"));
        GraphExecutors.Register(typeof(BindingAllocator), _ => new RecordingExecutor(binds: true, s_log));
        GraphExecutors.Register(typeof(CachingAllocator), _ => throw new InvalidOperationException("no executor is made just to ask"));
    }

    [Fact]
    public void AResidentWeightCacheDropsItsCopyByCacheKey()
    {
        lock (s_log)
        {
            s_log.Clear();
            Assert.True(DeviceResidency.KeepsWeightCopies(new CachingAllocator()));
            DeviceResidency.ReleaseWeightCopies(new CachingAllocator(), (IntPtr)0x1000, (IntPtr)0x2000);
            Assert.Equal(new[] { "release 4096" }, s_log);

            // An executor the caller names (one that also bound the host blocks) is told as well.
            s_log.Clear();
            var named = new RecordingExecutor(binds: false, s_log);
            DeviceResidency.ReleaseWeightCopies(new CachingAllocator(), (IntPtr)0x1000, (IntPtr)0x2000, named);
            Assert.Equal(new[] { "release 4096", "invalidate 8192" }, s_log);
        }
    }

    [Fact]
    public void AnExecutorThatBindsHostPointersDropsItsCopyByAddress()
    {
        lock (s_log)
        {
            s_log.Clear();
            Assert.False(DeviceResidency.KeepsWeightCopies(new BindingAllocator()));
            DeviceResidency.ReleaseWeightCopies(new BindingAllocator(), (IntPtr)0x1000, (IntPtr)0x2000);
            Assert.Equal(new[] { "invalidate 8192" }, s_log);

            // Nothing to drop for an executor that binds nothing, unless the caller names it.
            s_log.Clear();
            DeviceResidency.InvalidateBound((IntPtr)0x2000, new RecordingExecutor(binds: false, s_log));
            Assert.Empty(s_log);
            DeviceResidency.Invalidate((IntPtr)0x2000, new RecordingExecutor(binds: false, s_log));
            Assert.Single(s_log);
        }
    }

    [Fact]
    public void ALeaseDropsTheCopiesBeforeTheBytesAndOnlyOnce()
    {
        lock (s_log)
        {
            s_log.Clear();
            var allocator = new CpuAllocator(BlasEnum.DotNet);
            var tensor = new Tensor(allocator, DType.Float32, 16);
            IntPtr data = TensorComputePrimitives.GetStoragePointer(tensor);
            var executor = new RecordingExecutor(binds: true, s_log);
            var lease = new ResidentTensorLease(tensor, executor);
            Assert.Same(tensor, lease.Tensor);
            lease.Dispose();
            Assert.Equal(new[] { $"invalidate {data}" }, s_log);
            Assert.Null(lease.Tensor);
            lease.Dispose();   // a second release does nothing
            Assert.Single(s_log);
        }
    }

    /// <summary>The address-reuse hazard the release protocol closes, on a device that keeps copies by address: a weight is
    /// made resident, released through DeviceResidency, and new bytes are written at the same address; the next upload
    /// carries the new bytes, never the stale copy.</summary>
    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void AReleasedAddressBindsTheNewBytesOnTheDevice()
    {
        GgmlGraphExecutor executor = GgmlGraphExecutor.For(GgmlBackendType.Cuda);
        const int floats = 64 * 1024;
        long bytes = (long)floats * sizeof(float);
        IntPtr host = Marshal.AllocHGlobal((IntPtr)bytes);
        try
        {
            float[] first = Values(floats, 1), second = Values(floats, 2);
            Marshal.Copy(first, 0, host, floats);
            var weight = new Weight(host, floats, 1, bytes);
            Assert.Equal(1, executor.Preload(new List<IGraphWeight> { weight }, "device-residency-test"));
            Assert.True(GgmlBasicOps.HasResidentDeviceCopy(host, bytes));

            DeviceResidency.ReleaseWeightCopies(null, IntPtr.Zero, host, executor);
            Assert.False(GgmlBasicOps.HasResidentDeviceCopy(host, bytes), "the release left the old device copy");

            Marshal.Copy(second, 0, host, floats);
            Assert.Equal(1, executor.Preload(new List<IGraphWeight> { weight }, "device-residency-test"));
            Assert.Equal(second, Download(host, floats));
        }
        finally
        {
            executor.Invalidate(host);
            Marshal.FreeHGlobal(host);
        }
    }

    private sealed record Weight(IntPtr Data, long Ne0, long Ne1, long Bytes) : IGraphWeight
    {
        public int Format => 0;   // F32
        public long Ne2 => 1;
        public IntPtr CacheKey => Data;
    }

    private static float[] Values(int count, int seed)
    {
        var rng = new Random(seed);
        var v = new float[count];
        for (int i = 0; i < count; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    private static float[] Download(IntPtr key, int count)
    {
        var values = new float[count];
        var pin = GCHandle.Alloc(values, GCHandleType.Pinned);
        try
        {
            long length = (long)count * sizeof(float);
            int copied = GgmlBasicOps.DownloadDeviceCopyRanges(key, pin.AddrOfPinnedObject(),
                new long[] { 0 }, new long[] { 0 }, new long[] { length });
            Assert.Equal(1, copied);
            return values;
        }
        finally { pin.Free(); }
    }
}
