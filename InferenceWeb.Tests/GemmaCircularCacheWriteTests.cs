// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class GemmaCircularCacheWriteTests
{
    [Theory]
    [InlineData(DType.Float32, 647, 512, 139, 3)]
    [InlineData(DType.Float16, 647, 512, 139, 3)]
    [InlineData(DType.Float32, 1301, 512, 37, 2)]
    [InlineData(DType.Float16, 1301, 512, 37, 2)]
    [InlineData(DType.Float32, 19, 17, 9, 1)]
    [InlineData(DType.Float16, 19, 17, 9, 1)]
    [InlineData(DType.Float32, 17, 512, 509, 2)]
    [InlineData(DType.Float16, 17, 512, 509, 2)]
    [InlineData(DType.Float32, 37, 512, 500, 3)]
    [InlineData(DType.Float16, 37, 512, 500, 3)]
    [InlineData(DType.Float32, 2, 5, int.MaxValue, 2)]
    [InlineData(DType.Float16, 2, 5, int.MaxValue, 2)]
    [InlineData(DType.Float32, 2, 5, int.MaxValue, 33)]
    [InlineData(DType.Float16, 2, 5, int.MaxValue, 33)]
    public void HostCircularWritesMatchSequentialLastWriterForEveryPhysicalSlot(
        DType dtype, int rows, int capacity, int start, int heads)
    {
        const int width = 37;
        var model = (Gemma4Model)RuntimeHelpers.GetUninitializedObject(typeof(Gemma4Model));
        GC.SuppressFinalize(model);
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        // The copy's cache-invalidation tail consults the backend execution plan.
        // Initialize the CPU environment omitted by the weight-free fixture.
        const BindingFlags fields = BindingFlags.Instance | BindingFlags.NonPublic;
        typeof(ModelBase).GetField("_allocator", fields)!.SetValue(model, allocator);
        typeof(ModelBase).GetField("<ExecutionPlan>k__BackingField", fields)!
            .SetValue(model, new BackendExecutionPlan(BackendType.Cpu));
        using var source = new Tensor(allocator, DType.Float32, heads, rows, width);
        using var cache = new Tensor(allocator, dtype, heads, capacity, width);
        var copy = typeof(Gemma4Model).GetMethod("CopyToCacheCircular", BindingFlags.Instance | BindingFlags.NonPublic)!;
        float[] values = new float[heads * rows * width];
        for (int iteration = 0; iteration < 8; iteration++)
        {
            for (int h = 0; h < heads; h++)
            for (int row = 0; row < rows; row++)
            for (int d = 0; d < width; d++)
                values[(h * rows + row) * width + d] = h * 4 + row / 4f + d / 64f + iteration / 8f;
            Marshal.Copy(values, 0, source.Storage.PtrAtElement(0), values.Length);
            float[] expected = Enumerable.Repeat(-17.5f, heads * capacity * width).ToArray();
            CopyCache(cache, expected);
            // Reference applies every logical write sequentially, so it also
            // checks untouched slots, wrap points and original per-head stride.
            for (int row = 0; row < rows; row++)
            for (int h = 0; h < heads; h++)
            for (int d = 0; d < width; d++)
            {
                float value = values[(h * rows + row) * width + d];
                int slot = (int)(((long)start + row) % capacity);
                expected[(h * capacity + slot) * width + d]
                    = dtype == DType.Float16 ? (float)(System.Half)value : value;
            }
            copy.Invoke(model, new object[] { cache, source, start, rows, capacity });
            float[] actual = ReadCache(cache, expected.Length);
            Assert.Equal(expected, actual);
        }
    }

    private static void CopyCache(Tensor cache, float[] values)
    {
        byte[] bytes = cache.ElementType == DType.Float16
            ? MemoryMarshal.AsBytes(values.Select(v => (System.Half)v).ToArray().AsSpan()).ToArray()
            : MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
        Marshal.Copy(bytes, 0, cache.Storage.PtrAtElement(0), bytes.Length);
    }

    private static float[] ReadCache(Tensor cache, int elements)
    {
        byte[] bytes = new byte[elements * (cache.ElementType == DType.Float16 ? 2 : 4)];
        Marshal.Copy(cache.Storage.PtrAtElement(0), bytes, 0, bytes.Length);
        if (cache.ElementType == DType.Float32) return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
        return MemoryMarshal.Cast<byte, System.Half>(bytes).ToArray().Select(v => (float)v).ToArray();
    }
}
