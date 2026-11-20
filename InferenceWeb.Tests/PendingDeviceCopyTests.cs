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
using TensorSharp.GGML;
using TensorSharp.Graph;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// A weight upload that is abandoned after its weights were bound must not leave their device copies behind unwritten.
/// The device-copy cache publishes a new copy, marked initialised, before any byte reaches it, and a later bind of the
/// same host bytes is a cache hit that uploads nothing; an entry point that queued its uploads and then returned early
/// used to leave every later graph reading uninitialised device memory as the weights (on the RTX A4500, one MiniMax-H3
/// request that failed for VRAM turned every later picture of the server into noise). The graph builder's weight
/// preload is such an entry point: its binds come first, its uploads after the whole list was read, and a malformed
/// entry in the list throws in between. The copies a failed preload made are invalidated, so the next preload (or the
/// first graph) uploads them, and the device holds the host bytes.
/// </summary>
public sealed class PendingDeviceCopyTests
{
    private const int F32 = 0;   // GGML_TYPE_F32
    private const int Floats = 64 * 1024;

    private sealed record Weight(IntPtr Data, long Ne0, long Ne1, long Bytes) : IGraphWeight
    {
        public int Format => F32;
        public long Ne2 => 1;
        public IntPtr CacheKey => Data;
    }

    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void AFailedPreloadLeavesNoUnwrittenDeviceCopy()
    {
        GgmlGraphExecutor executor = GgmlGraphExecutor.For(GgmlBackendType.Cuda);
        long bytes = (long)Floats * sizeof(float);
        IntPtr a = Marshal.AllocHGlobal((IntPtr)bytes), b = Marshal.AllocHGlobal((IntPtr)bytes);
        try
        {
            float[] va = Values(Floats, 1), vb = Values(Floats, 2);
            Marshal.Copy(va, 0, a, Floats);
            Marshal.Copy(vb, 0, b, Floats);
            var wa = new Weight(a, Floats, 1, bytes);
            var wb = new Weight(b, Floats, 1, bytes);
            // Fewer host bytes than its tensor: the native preload throws on it, after binding the two before it.
            var malformed = new Weight(b + 64, 1024, 1, 16);

            var error = Assert.Throws<InvalidOperationException>(() =>
                executor.Preload(new List<IGraphWeight> { wa, wb, malformed }, "pending-device-copy-test"));
            Assert.Contains("fewer host bytes", error.Message);
            Assert.False(GgmlBasicOps.HasResidentDeviceCopy(a, bytes), "the failed preload left a device copy it never wrote");
            Assert.False(GgmlBasicOps.HasResidentDeviceCopy(b, bytes), "the failed preload left a device copy it never wrote");

            // The next preload uploads both, and the device holds the host bytes.
            Assert.Equal(2, executor.Preload(new List<IGraphWeight> { wa, wb }, "pending-device-copy-test"));
            Assert.Equal(va, Download(a, Floats));
            Assert.Equal(vb, Download(b, Floats));
            Assert.Equal(0, executor.Preload(new List<IGraphWeight> { wa, wb }, "pending-device-copy-test"));
        }
        finally
        {
            executor.Invalidate(a);
            executor.Invalidate(b);
            Marshal.FreeHGlobal(a);
            Marshal.FreeHGlobal(b);
        }
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
        }
        finally
        {
            pin.Free();
        }
        return values;
    }
}
