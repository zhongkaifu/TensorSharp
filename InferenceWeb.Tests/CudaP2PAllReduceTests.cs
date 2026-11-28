// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

/// <summary>The local tensor-parallel group's all-reduce over two GPUs sums exactly at every size a decoder hands it
/// (one decode row of a small model's hidden size up to a prompt chunk), repeatedly, with work queued before it.</summary>
public sealed class CudaP2PAllReduceTests
{
    [CudaFact]
    public void TwoGpuAllReduceSumsEverySize()
    {
        // CudaFact covers availability; the two-device requirement stays a silent gate (no lane filters on device count).
        if (CudaDevice.GetDeviceCount() < 2)
            return;
        using var group = new TensorParallelGroup(2);
        var random = new Random(7);
        foreach (int n in new[] { 1024, 2048, 4096, 1024 * 3, 1024 * 32, 1024 * 1826, 4096 * 1826 })
        {
            for (int rep = 0; rep < 4; rep++)
            {
                var a = new float[n];
                var b = new float[n];
                for (int i = 0; i < n; i++)
                {
                    a[i] = (float)(random.NextDouble() - 0.5);
                    b[i] = (float)(random.NextDouble() - 0.5);
                }
                using var ta = new Tensor(group.GetAllocator(0), DType.Float32, n);
                using var tb = new Tensor(group.GetAllocator(1), DType.Float32, n);
                ta.SetElementsAsFloat(a);
                tb.SetElementsAsFloat(b);
                group.AllReduce(new[] { ta, tb });
                float[] ra = ta.GetElementsAsFloat(n), rb = tb.GetElementsAsFloat(n);
                int bad = 0, first = -1;
                for (int i = 0; i < n; i++)
                {
                    float want = a[i] + b[i];
                    if (ra[i] != want || rb[i] != want)
                    {
                        if (first < 0) first = i;
                        bad++;
                    }
                }
                Assert.True(bad == 0,
                    $"n={n} rep={rep}: {bad} wrong sums, first at {first}: rank0 {(first >= 0 ? ra[first] : 0)} rank1 {(first >= 0 ? rb[first] : 0)} want {(first >= 0 ? a[first] + b[first] : 0)}");
            }
        }
    }

    /// <summary>
    /// The device-ordered reduction (event waits between the ranks' streams, no host synchronization) leaves every rank the
    /// bits the host-synchronized reduce-to-zero leaves, through a chain like a decode step's: each round every rank scales
    /// its copy of the stream, adds its own term and reduces, with no host read in between, so a reduction ordered wrongly
    /// on the devices would read a partial before its rank wrote it, or overwrite one a peer still reads.
    /// </summary>
    [CudaFact]
    public void DeviceOrderedAllReduceMatchesTheHostSynchronizedOne()
    {
        if (CudaDevice.GetDeviceCount() < 2)
            return;
        using var group = new TensorParallelGroup(2);
        if (!group.Communicator.DeviceOrdered)
            return;   // a pair without verified peer reads keeps the host-synchronized reduction
        foreach (int n in new[] { 1024, 4096, 1024 * 37, 4096 * 512 })
        {
            float[][] device = Chain(group, n, hostSynchronized: false), host = Chain(group, n, hostSynchronized: true);
            Assert.True(device[0].AsSpan().SequenceEqual(device[1]), $"n={n}: the ranks' sums differ");
            Assert.True(device[0].AsSpan().SequenceEqual(host[0]), $"n={n}: the device-ordered sums differ from the host-synchronized ones");
        }
    }

    private static float[][] Chain(TensorParallelGroup group, int n, bool hostSynchronized)
    {
        const int rounds = 24;
        group.Communicator.HostSynchronizedForTests = hostSynchronized;
        try
        {
            var random = new Random(11);
            var x = new Tensor[2];
            var terms = new Tensor[2, rounds];
            try
            {
                for (int r = 0; r < 2; r++)
                {
                    x[r] = new Tensor(group.GetAllocator(r), DType.Float32, n);
                    x[r].SetElementsAsFloat(new float[n]);
                    for (int k = 0; k < rounds; k++)
                    {
                        var values = new float[n];
                        for (int i = 0; i < n; i++) values[i] = (float)(random.NextDouble() - 0.5);
                        terms[r, k] = new Tensor(group.GetAllocator(r), DType.Float32, n);
                        terms[r, k].SetElementsAsFloat(values);
                    }
                }
                for (int k = 0; k < rounds; k++)
                {
                    for (int r = 0; r < 2; r++)
                    {
                        Ops.Mul(x[r], x[r], 0.5f);
                        Ops.Add(x[r], x[r], terms[r, k]);
                    }
                    group.AllReduce(x);
                }
                return new[] { x[0].GetElementsAsFloat(n), x[1].GetElementsAsFloat(n) };
            }
            finally
            {
                foreach (Tensor t in x) t?.Dispose();
                foreach (Tensor t in terms) t?.Dispose();
            }
        }
        finally
        {
            group.Communicator.HostSynchronizedForTests = false;
        }
    }
}
