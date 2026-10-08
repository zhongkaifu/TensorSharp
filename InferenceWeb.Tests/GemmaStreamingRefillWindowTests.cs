// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

public sealed class GemmaStreamingRefillWindowTests
{
    [Theory]
    [InlineData(DType.Float32, 256, true)]
    [InlineData(DType.Float32, 512, true)]
    [InlineData(DType.Float32, 777, true)]
    [InlineData(DType.Float16, 256, true)]
    [InlineData(DType.Float16, 512, true)]
    [InlineData(DType.Float16, 777, true)]
    [InlineData(DType.Float32, 256, false)]
    [InlineData(DType.Float32, 512, false)]
    [InlineData(DType.Float32, 777, false)]
    [InlineData(DType.Float16, 256, false)]
    [InlineData(DType.Float16, 512, false)]
    [InlineData(DType.Float16, 777, false)]
    public void SharedDonorGatherPreservesResidentGeometryAndExactlyTheCausalWindow(
        DType dtype, int startPos, bool streaming)
    {
        const int capacity = 512, heads = 2, dim = 3, chunk = 132;
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        using var k = new Tensor(allocator, dtype, heads, capacity, dim);
        using var v = new Tensor(allocator, dtype, heads, capacity, dim);
        float[] physical = Enumerable.Repeat(-1f, heads * capacity * dim).ToArray();
        for (int position = 0; position < startPos; position++)
        for (int head = 0; head < heads; head++)
        for (int d = 0; d < dim; d++)
            physical[(head * capacity + position % capacity) * dim + d] = position;
        Copy(k, physical); Copy(v, physical.Select(x => -x).ToArray());
        var model = (Gemma4Model)RuntimeHelpers.GetUninitializedObject(typeof(Gemma4Model));
        GC.SuppressFinalize(model);
        Set(typeof(ModelBase), model, "_allocator", allocator);
        Set(typeof(ModelBase), model, "<Config>k__BackingField", new ModelConfig { NumLayers = 2, NumKVHeads = heads });
        if (streaming)
        {
            var budget = new MemoryBudget(new[] { new MemoryCharge("host", 1 << 20), new MemoryCharge("gpu", 2 << 20) });
            Set(typeof(ModelBase), model, "WeightStreaming", new WeightStreamingOptions(budget, "host", new[] { "gpu" }));
        }
        Set(typeof(Gemma4Model), model, "_slidingWindow", capacity);
        Set(typeof(Gemma4Model), model, "_localHeadDim", dim);
        Set(typeof(Gemma4Model), model, "_slidingWindowPattern", new[] { true, true });
        Set(typeof(Gemma4Model), model, "_kvDonorMap", new Dictionary<int, int> { [1] = 0 });
        Set(typeof(Gemma4Model), model, "_kvCacheK", new[] { k, k });
        Set(typeof(Gemma4Model), model, "_kvCacheV", new[] { v, v });
        Set(typeof(Gemma4Model), model, "_kvCacheSize", new[] { capacity, capacity });
        try
        {
            Invoke(model, "PrepareSwaPrevWindowsForChunk", startPos, chunk);
            var gathered = (Dictionary<int, (Tensor k, Tensor v)>)typeof(Gemma4Model)
                .GetField("_swaPrevWindow", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
            Assert.Equal(0, Assert.Single(gathered).Key); // Shared layer borrows one donor snapshot.
            int previous = startPos < capacity ? startPos : (streaming ? capacity : capacity - 1);
            Assert.Equal(previous, gathered[0].k.Sizes[1]);
            float[] captured = gathered[0].k.GetElementsAsFloat(heads * previous * dim);
            float[] capturedV = gathered[0].v.GetElementsAsFloat(captured.Length);
            for (int head = 0; head < heads; head++)
            for (int row = 0; row < previous; row++)
            for (int d = 0; d < dim; d++)
            {
                int index = (head * previous + row) * dim + d;
                Assert.Equal(startPos - previous + row, captured[index]);
                Assert.Equal(-captured[index], capturedV[index]);
            }
            // Replay the attention mask over the actual gathered row values
            // plus fresh logical rows, and compare to an absolute-position
            // oracle independent of buffer offsets and masked leading slots.
            float[] keys = Enumerable.Range(0, previous).Select(i => captured[i * dim])
                .Concat(Enumerable.Range(startPos, chunk).Select(i => (float)i)).ToArray();
            for (int query = 0; query < chunk; query++)
            {
                int cutoff = keys.Length - chunk + query;
                float[] attended = keys.Where((_, index) => index <= cutoff && index >= Math.Max(0, cutoff - capacity + 1)).ToArray();
                int first = Math.Max(0, startPos + query - capacity + 1);
                Assert.Equal(Enumerable.Range(first, startPos + query - first + 1).Select(i => (float)i).ToArray(), attended);
            }
            if (streaming && startPos >= capacity)
                Assert.Equal(startPos - capacity, keys[0]); // Retained for geometry; masked from every query.
            Copy(k, new float[physical.Length]); Copy(v, new float[physical.Length]);
            Assert.Equal(captured, gathered[0].k.GetElementsAsFloat(captured.Length));
            Assert.Equal(capturedV, gathered[0].v.GetElementsAsFloat(capturedV.Length));
        }
        finally { Invoke(model, "DisposeSwaPrevWindows"); }
    }

    private static void Set(Type type, object target, string name, object value)
        => type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(target, value);

    private static void Invoke(object target, string method, params object[] arguments)
        => typeof(Gemma4Model).GetMethod(method, BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(target, arguments);

    private static void Copy(Tensor tensor, float[] values)
    {
        byte[] bytes = tensor.ElementType == DType.Float16
            ? MemoryMarshal.AsBytes(values.Select(v => (System.Half)v).ToArray().AsSpan()).ToArray()
            : MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
        Marshal.Copy(bytes, 0, tensor.Storage.PtrAtElement(0), bytes.Length);
    }
}
