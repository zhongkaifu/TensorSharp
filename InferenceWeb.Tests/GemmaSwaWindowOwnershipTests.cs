// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Cpu;

namespace InferenceWeb.Tests;

public sealed class GemmaSwaWindowOwnershipTests
{
    [Fact]
    public void FailedConcatenationReleasesResultWithoutReleasingInputs()
    {
        var cpu = new CpuAllocator(BlasEnum.DotNet);
        using var previous = new Tensor(cpu, DType.Float32, 2, 7, 3);
        using var fresh = new Tensor(cpu, DType.Float32, 2, 5, 3);
        var allocator = new FailingAllocator(1, failAfterAllocation: true);
        var model = (Gemma4Model)RuntimeHelpers.GetUninitializedObject(typeof(Gemma4Model));
        GC.SuppressFinalize(model);
        typeof(ModelBase).GetField("_allocator", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(model, allocator);
        var concat = typeof(Gemma4Model).GetMethod("ConcatHeadFirstKV", BindingFlags.Instance | BindingFlags.NonPublic)!;
        var error = Assert.Throws<TargetInvocationException>(() => concat.Invoke(model, new object[] { previous, fresh }));
        Assert.Equal("Injected SWA window failure.", error.InnerException!.Message);
        Assert.Equal(IntPtr.Zero, Assert.Single(allocator.Created).buffer);
        Assert.NotEqual(IntPtr.Zero, ((CpuStorage)previous.Storage).buffer);
        Assert.NotEqual(IntPtr.Zero, ((CpuStorage)fresh.Storage).buffer);
    }

    [Theory]
    [InlineData(DType.Float32, 2, false)]
    [InlineData(DType.Float32, 2, true)]
    [InlineData(DType.Float32, 4, false)]
    [InlineData(DType.Float32, 4, true)]
    [InlineData(DType.Float16, 2, false)]
    [InlineData(DType.Float16, 2, true)]
    [InlineData(DType.Float16, 4, false)]
    [InlineData(DType.Float16, 4, true)]
    public void FailedValueWindowReleasesUnpublishedOwnersAndPreservesPublishedPairs(
        DType dtype, int failAt, bool failAfterAllocation)
    {
        const int capacity = 17, heads = 2, dim = 3;
        var cpu = new CpuAllocator(BlasEnum.DotNet);
        using var k = new Tensor(cpu, dtype, heads, capacity, dim);
        using var v = new Tensor(cpu, dtype, heads, capacity, dim);
        // A host-read failure happens after the temporary tensor is allocated,
        // exercising BuildSwaPrevWindow's own ownership boundary too.
        var allocator = new FailingAllocator(failAt, failAfterAllocation);
        var model = (Gemma4Model)RuntimeHelpers.GetUninitializedObject(typeof(Gemma4Model));
        GC.SuppressFinalize(model);
        Set(typeof(ModelBase), "_allocator", allocator);
        Set(typeof(ModelBase), "<Config>k__BackingField", new ModelConfig { NumLayers = 2, NumKVHeads = heads });
        Set(typeof(Gemma4Model), "_slidingWindow", capacity);
        Set(typeof(Gemma4Model), "_localHeadDim", dim);
        Set(typeof(Gemma4Model), "_slidingWindowPattern", new[] { true, true });
        Set(typeof(Gemma4Model), "_kvDonorMap", new Dictionary<int, int>());
        Set(typeof(Gemma4Model), "_kvCacheK", new[] { k, k });
        Set(typeof(Gemma4Model), "_kvCacheV", new[] { v, v });
        Set(typeof(Gemma4Model), "_kvCacheSize", new[] { capacity, capacity });
        try
        {
            var error = Assert.Throws<TargetInvocationException>(() => Invoke("PrepareSwaPrevWindowsForChunk", 23, 5));
            Assert.Equal("Injected SWA window failure.", error.InnerException!.Message);
            Assert.Equal(failAt, allocator.Attempts);
            Assert.Equal(failAt - (failAfterAllocation ? 0 : 1), allocator.Created.Count);
            var windows = (Dictionary<int, (Tensor k, Tensor v)>)typeof(Gemma4Model)
                .GetField("_swaPrevWindow", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
            int published = failAt == 4 ? 2 : 0;
            Assert.Equal(published / 2, windows.Count);
            for (int i = 0; i < allocator.Created.Count; i++)
            {
                // No finalizers or forced GC: failed temporaries must already
                // have returned their physical CPU allocations synchronously.
                if (i < published) Assert.NotEqual(IntPtr.Zero, allocator.Created[i].buffer);
                else Assert.Equal(IntPtr.Zero, allocator.Created[i].buffer);
            }
            if (published != 0)
            {
                Assert.Same(allocator.Created[0], windows[0].k.Storage);
                Assert.Same(allocator.Created[1], windows[0].v.Storage);
            }
            Invoke("DisposeSwaPrevWindows");
            Assert.All(allocator.Created, storage => Assert.Equal(IntPtr.Zero, storage.buffer));
            Assert.NotEqual(IntPtr.Zero, ((CpuStorage)k.Storage).buffer);
            Assert.NotEqual(IntPtr.Zero, ((CpuStorage)v.Storage).buffer);
        }
        finally { Invoke("DisposeSwaPrevWindows"); }

        void Set(Type type, string name, object value)
            => type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, value);
        void Invoke(string method, params object[] arguments)
            => typeof(Gemma4Model).GetMethod(method, BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(model, arguments);
    }

    private sealed class FailingAllocator(int failAt, bool failAfterAllocation) : IAllocator
    {
        public int Attempts { get; private set; }
        public List<CpuStorage> Created { get; } = [];
        public BlasEnum BlasEnum => BlasEnum.DotNet;
        public int DeviceId => 0;
        public float GetAllocatedMemoryRatio() => 0;
        public Storage Allocate(DType type, long count)
        {
            bool fail = ++Attempts == failAt;
            if (fail && !failAfterAllocation) throw new OutOfMemoryException("Injected SWA window failure.");
            var storage = new FailingStorage(this, type, count, fail);
            Created.Add(storage);
            return storage;
        }
    }

    private sealed class FailingStorage(IAllocator allocator, DType type, long count, bool failOnRead)
        : CpuStorage(allocator, type, count)
    {
        public override void EnsureHostReadable()
        {
            if (failOnRead) throw new InvalidOperationException("Injected SWA window failure.");
        }
    }
}
