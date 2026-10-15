// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public class CudaCircularCacheWriteTests
{
    [CudaTheory]
    [InlineData(DType.Float32, 647, 512, 0)]
    [InlineData(DType.Float16, 647, 512, 0)]
    [InlineData(DType.Float32, 647, 512, 37)]
    [InlineData(DType.Float16, 647, 512, 37)]
    [InlineData(DType.Float32, 1295, 512, 1041)]
    [InlineData(DType.Float16, 1295, 512, 1041)]
    [InlineData(DType.Float32, 4, 5, 3)]
    [InlineData(DType.Float16, 4, 5, 3)]
    [InlineData(DType.Float32, 1, 5, 13)]
    [InlineData(DType.Float16, 1, 5, 13)]
    [InlineData(DType.Float32, 2, 5, int.MaxValue)]
    [InlineData(DType.Float16, 2, 5, int.MaxValue)]
    public void OversizedAndWrappingWrites_MatchSequentialRing_WithGuardsAndUnchangedSource(
        DType cacheType, int sequence, int capacity, int start)
    {
        const int heads = 3, width = 5, guard = 17;
        const float sentinel = -4096;
        int sourceCount = heads * sequence * width, cacheCount = heads * capacity * width;
        using var allocator = new CudaAllocator();
        using var sourceBacking = new Tensor(allocator, DType.Float32, sourceCount + 2 * guard);
        using var sourceRegion = sourceBacking.Narrow(0, guard, sourceCount);
        using var source = sourceRegion.View(heads, sequence, width);
        using var cacheBacking = new Tensor(allocator, cacheType, cacheCount + 2 * guard);
        using var cacheRegion = cacheBacking.Narrow(0, guard, cacheCount);
        using var cache = cacheRegion.View(heads, capacity, width);
        // A ring oracle performs ordered logical writes, independent of the
        // production kernel's thread mapping or its retained-prefix filter.
        for (int revision = 0; revision < 8; revision++)
        {
            float[] sourceValues = Enumerable.Repeat(sentinel, sourceCount + 2 * guard).ToArray();
            float[] expected = Enumerable.Repeat(sentinel, cacheCount + 2 * guard).ToArray();
            for (int head = 0; head < heads; head++)
                for (int seq = 0; seq < sequence; seq++)
                    for (int dim = 0; dim < width; dim++)
                    {
                        float value = head * 1000 + seq * .5f + dim * .0625f + revision * .25f;
                        sourceValues[guard + (head * sequence + seq) * width + dim] = value;
                        int slot = (int)(((long)start + seq) % capacity);
                        expected[guard + (head * capacity + slot) * width + dim] =
                            cacheType == DType.Float16 ? (float)(System.Half)value : value;
                    }
            sourceBacking.SetElementsAsFloat(sourceValues);
            Ops.Fill(cacheBacking, sentinel);
            if (start == int.MaxValue)
                Assert.False(CudaFusedOps.TryCopyHeadFirstToCache(cache, source, start, sequence, capacity, circular: false));
            Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(cache, source, start, sequence, capacity, circular: true));
            Assert.Equal(expected, ReadAll(cacheBacking));
            // Force an actual device download even though the source is supposed
            // to be immutable, so a stale clean host mirror cannot hide a write.
            Assert.True(CudaFusedOps.TryMarkDeviceModified(sourceBacking));
            Assert.Equal(sourceValues, sourceBacking.GetElementsAsFloat(sourceValues.Length));
        }
    }

    [CudaTheory]
    [InlineData(DType.Float32)]
    [InlineData(DType.Float16)]
    public void SingleTokenGraphReplay_UsesFreshCircularWritePosition(DType cacheType)
    {
        const int heads = 2, width = 3, capacity = 5;
        using var allocator = new CudaAllocator();
        using var graph = new CudaPrefillGraphCache(allocator);
        using var dynamic = new CudaDecodeDynParams(allocator);
        Assert.True(graph.IsUsable);
        Assert.True(dynamic.IsValid);
        using var cache = new Tensor(allocator, cacheType, heads, capacity, width);
        using var source = new Tensor(allocator, DType.Float32, heads, 1, width);
        using var replayInput = new Tensor(allocator, DType.Float32, 1, width);
        Ops.Fill(cache, -4);
        source.SetElementsAsFloat([.1f, .2f, .3f, 1.1f, 1.2f, 1.3f]);
        replayInput.SetElementsAsFloat(new float[width]);
        foreach (var tensor in new[] { cache, source, replayInput })
            Assert.True(CudaFusedOps.TryEnsureDeviceResident(tensor));
        // Load the kernel/module before entering capture, then restore the ring.
        Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(cache, source, 0, 1, capacity, circular: true));
        Ops.Fill(cache, -4);
        const string key = "circular-cache-write-single-token";
        Assert.False(graph.ShouldCapture(key));
        Assert.True(graph.ShouldCapture(key));
        dynamic.Write(attendLen: 1, kvWritePos: 3, convWriteIdx: 0, ropePos: 3);
        Assert.True(graph.BeginCapture(key));
        try
        {
            dynamic.EnqueueUpload();
            dynamic.Activate();
            Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(cache, source, 0, 1, capacity, circular: true));
            CudaDecodeDynParams.Deactivate();
            Assert.True(graph.EndCaptureAndLaunch(key, replayInput, null, maxAttendLen: capacity));
        }
        finally { CudaDecodeDynParams.Deactivate(); }

        float[] expected = Enumerable.Repeat(-4f, heads * capacity * width).ToArray();
        UpdateExpected(3, [.1f, .2f, .3f, 1.1f, 1.2f, 1.3f]);
        Assert.Equal(expected, ReadAll(cache));
        foreach (int position in new[] { 7, 10, 14 })
        {
            float[] values = Enumerable.Range(0, heads * width).Select(i => position + i * .125f).ToArray();
            source.SetElementsAsFloat(values);
            Assert.True(CudaFusedOps.TryEnsureDeviceResident(source));
            dynamic.Write(attendLen: 1, kvWritePos: position, convWriteIdx: 0, ropePos: position);
            graph.Replay(key);
            Assert.True(CudaFusedOps.TryMarkDeviceModified(cache));
            UpdateExpected(position, values);
            Assert.Equal(expected, ReadAll(cache));
        }
        void UpdateExpected(int position, float[] values)
        {
            for (int head = 0; head < heads; head++) for (int dim = 0; dim < width; dim++)
            {
                float value = values[head * width + dim];
                expected[(head * capacity + position % capacity) * width + dim] =
                    cacheType == DType.Float16 ? (float)(System.Half)value : value;
            }
        }
    }

    private static float[] ReadAll(Tensor tensor)
    {
        int count = checked((int)tensor.ElementCount());
        if (tensor.ElementType == DType.Float32) return tensor.GetElementsAsFloat(count);
        // CUDA's bulk Float32 accessor does not convert F16. The storage scalar
        // accessor synchronizes once and then reads the same downloaded mirror.
        return Enumerable.Range(0, count)
            .Select(i => tensor.Storage.GetElementAsFloat(tensor.StorageOffset + i)).ToArray();
    }
}
