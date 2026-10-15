// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>Calls the actual production embedding/broadcast path with CPU
/// pointers, the real GGML host pool and real allocation credit. No model file,
/// native library, GPU, forced GC or synthetic reservation bypass is involved.</summary>
[Collection(EngineEnvironmentCollection.Name)]
public sealed class Qwen4ExpEmbeddingOwnershipTests
{
    [Fact]
    public void ResidualQuotaRefusalReleasesTheReusedWideEmbedding()
    {
        using var f = new Fixture();
        f.SeedWideEmbedding();
        using var envelope = f.Budget.Reserve([new("ram", 0)]);
        Exception error;
        using (f.Scope.EnterExecution(envelope)) error = f.Forward();
        Assert.IsType<MemoryPressureException>(error);
        Assert.Contains("Host allocation of ", error.Message);
        Assert.Contains("exceeds its allocation envelope.", error.Message);
        Assert.Equal(5_242_880, f.ReusedEmbeddingBytes);
        f.ReleasePublishedAndClosePool();
        f.AssertReleased();
    }

    [Fact]
    public void VisionReadFailureReleasesEmbeddingAndPendingVisionAndKeepsOriginalError()
    {
        using var f = new Fixture();
        var failure = new IOException("vision host read refused");
        f.AddVision(failure);
        f.SeedWideEmbedding();
        using var envelope = f.Budget.Reserve([new("ram", 0)]);
        Exception error;
        using (f.Scope.EnterExecution(envelope)) error = f.Forward();
        Assert.Same(failure, error);
        Assert.Empty(f.Vision);
        f.ReleasePublishedAndClosePool();
        f.AssertReleased();
    }

    [Fact]
    public void BroadcastReadFailureReleasesBothEmbeddingAndUnpublishedResidual()
    {
        using var f = new Fixture();
        f.SeedWideEmbedding();
        var failure = new IOException("residual host read refused");
        f.Allocator.OnStorageCreated = storage =>
        {
            if (storage.ElementCount == 3L * Fixture.Hidden * 4) storage.ReadFailure = failure;
        };
        using var envelope = f.Budget.Reserve([new("ram", 1 << 20)]);
        Exception error;
        using (f.Scope.EnterExecution(envelope)) error = f.Forward();
        Assert.Same(failure, error);
        envelope.Dispose();
        f.ReleasePublishedAndClosePool();
        f.AssertReleased();
    }

    [Fact]
    public void SuccessfulBroadcastCopiesEveryStreamAndReturnsALiveOwnedResult()
    {
        using var f = new Fixture();
        using var input = new Tensor(f.Allocator, DType.Float32, 3, Fixture.Hidden);
        var values = Enumerable.Range(0, 3 * Fixture.Hidden).Select(i => i * 0.25f - 1000).ToArray();
        Marshal.Copy(values, 0, input.Storage.PtrAtElement(0), values.Length);
        using var result = f.Broadcast(input);
        var actual = new float[3 * Fixture.Hidden * 4];
        Marshal.Copy(result.Storage.PtrAtElement(0), actual, 0, actual.Length);
        for (int row = 0; row < 3; row++)
            for (int stream = 0; stream < 4; stream++)
                Assert.Equal(values.AsSpan(row * Fixture.Hidden, Fixture.Hidden).ToArray(),
                    actual.AsSpan((row * 4 + stream) * Fixture.Hidden, Fixture.Hidden).ToArray());
        f.ReleasePublishedAndClosePool();
        Assert.True(f.Scope.Usage.Bytes > 0); // Returning a result does not refund its physical bytes.
        result.Dispose();
        input.Dispose();
        f.AssertReleased();
    }

    private sealed class Fixture : IDisposable
    {
        internal const int Hidden = 2560;
        internal readonly MemoryBudget Budget = new([new("ram", 16L << 20)]);
        internal readonly HostAllocationBudgetScope Scope;
        internal readonly GgmlMemoryPool Pool = new(GgmlBackendType.Cuda);
        internal readonly Qwen4ExpModel Model = (Qwen4ExpModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen4ExpModel));
        internal readonly PoolAllocator Allocator;
        internal readonly List<(Tensor Embeddings, int StartPosition)> Vision = [];
        private readonly Tensor _weight;
        private bool _publishedReleased;
        internal long ReusedEmbeddingBytes;

        internal Fixture()
        {
            Scope = new(Budget, ["ram"]);
            Allocator = new(Pool);
            Set(typeof(ModelBase), "_backend", BackendType.Cpu);
            Set(typeof(ModelBase), "<ExecutionPlan>k__BackingField", new BackendExecutionPlan(BackendType.Cpu));
            Set(typeof(ModelBase), "_allocator", Allocator);
            Set(typeof(ModelBase), "_forwardSw", new Stopwatch());
            typeof(ModelBase).GetProperty(nameof(ModelBase.Config))!.SetValue(Model,
                new ModelConfig { Architecture = Qwen4ExpModel.ArchitectureId, HiddenSize = Hidden, NumLayers = 0 });
            foreach (string name in new[] { "_weights", "_quantWeights" })
            {
                var field = typeof(ModelBase).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!;
                field.SetValue(Model, Activator.CreateInstance(field.FieldType));
            }
            _weight = new Tensor(Allocator, DType.Float32, 2, Hidden);
            var weightValues = Enumerable.Range(0, 2 * Hidden).Select(i => i * 0.125f).ToArray();
            Marshal.Copy(weightValues, 0, _weight.Storage.PtrAtElement(0), weightValues.Length);
            ((Dictionary<string, Tensor>)typeof(ModelBase).GetField("_weights", BindingFlags.Instance | BindingFlags.NonPublic)!
                .GetValue(Model)!).Add("token_embd.weight", _weight);
            Set(typeof(Qwen4ExpModel), "_visionEmbeddingsList", Vision);
            Set(typeof(Qwen4ExpModel), "_hc", 4);
            Set(typeof(Qwen4ExpModel), "_hcDim", Hidden * 4);
            Set(typeof(Qwen4ExpModel), "_kvCacheCapacity", 4096);
        }

        internal void SeedWideEmbedding()
        {
            long before = Scope.Usage.Bytes;
            using (var seed = new Tensor(Allocator, DType.Float32, 512, Hidden))
                ReusedEmbeddingBytes = Scope.Usage.Bytes - before;
            Assert.Equal(5_242_880, ReusedEmbeddingBytes);
        }

        internal void AddVision(Exception failure)
        {
            var tensor = new Tensor(Allocator, DType.Float32, 1, Hidden);
            ((PoolStorage)tensor.Storage).ReadFailure = failure;
            Vision.Add((tensor, 0));
        }

        internal Exception Forward()
        {
            var method = typeof(Qwen4ExpModel).GetMethod("ForwardCore", BindingFlags.Instance | BindingFlags.NonPublic)!;
            return Assert.Throws<TargetInvocationException>(() => method.Invoke(Model, [new[] { 0, 1, 0 }])).InnerException!;
        }

        internal Tensor Broadcast(Tensor input)
            => (Tensor)typeof(Qwen4ExpModel).GetMethod("BroadcastToStreams", BindingFlags.Instance | BindingFlags.NonPublic)!
                .Invoke(Model, [input, 3])!;

        internal void ReleasePublishedAndClosePool()
        {
            if (!_publishedReleased)
            {
                _weight.Dispose();
                foreach (var (tensor, _) in Vision) tensor.Dispose();
                Vision.Clear();
                _publishedReleased = true;
            }
            Pool.Close(); // Deliberately cannot free outstanding, leaked tensors.
        }

        internal void AssertReleased()
        {
            Assert.Equal(0, Scope.Usage.Bytes);
            Assert.Equal(0, Scope.Usage.Allocations);
            Assert.Equal(0, Budget.Snapshot().Single().Committed);
            Assert.Equal(0, Budget.Snapshot().Single().Reserved);
        }

        private void Set(Type type, string name, object value)
            => type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(Model, value);

        public void Dispose()
        {
            ReleasePublishedAndClosePool();
            // Failed red assertions must not contaminate the process-wide host
            // scope. Dispose the actual remaining fixture storage only AFTER
            // the ownership assertions; never manually refund ledger credit.
            foreach (var storage in Allocator.Storages)
                storage.ReleaseIfLive();
            Pool.Close();
            Scope.Dispose();
        }
    }

    private sealed class PoolAllocator(GgmlMemoryPool pool) : IAllocator
    {
        public BlasEnum BlasEnum => BlasEnum.DotNet;
        public int DeviceId => 0;
        internal readonly List<PoolStorage> Storages = [];
        internal Action<PoolStorage>? OnStorageCreated;
        public Storage Allocate(DType elementType, long elementCount)
        {
            var storage = new PoolStorage(this, pool, elementType, elementCount);
            Storages.Add(storage);
            OnStorageCreated?.Invoke(storage);
            return storage;
        }
        public float GetAllocatedMemoryRatio() => 0;
    }

    // CPU-only pointer storage using the REAL budgeted GGML pool. The normal
    // GGML storage's host-read hook calls native code, which is not needed for
    // a deterministic host copy/failure ownership test.
    private sealed class PoolStorage : Storage
    {
        private readonly GgmlMemoryPool _pool;
        private IntPtr _pointer;
        internal Exception? ReadFailure;
        internal PoolStorage(IAllocator allocator, GgmlMemoryPool pool, DType dtype, long count) : base(allocator, dtype, count)
        { _pool = pool; _pointer = pool.Allocate(ByteLength); }
        protected override void Destroy()
        { if (_pointer != IntPtr.Zero) { _pool.Free(_pointer, ByteLength); _pointer = IntPtr.Zero; } }
        public override void EnsureHostReadable() { if (ReadFailure != null) throw ReadFailure; }
        internal void ReleaseIfLive() { if (GetCurrentRefCount() > 0) Release(); }
        public override IntPtr PtrAtElement(long index) => _pointer + checked((int)(index * ElementType.Size()));
        public override string LocationDescription() => "CPU actual budgeted host pool fixture";
        public override int[] GetElementsAsInt(long index, int length) => throw new NotSupportedException();
        public override void SetElementsAsInt(long index, int[] value) => throw new NotSupportedException();
        public override float GetElementAsFloat(long index) => throw new NotSupportedException();
        public override float[] GetElementsAsFloat(long index, int length) => throw new NotSupportedException();
        public override void SetElementAsFloat(long index, float value) => throw new NotSupportedException();
        public override void SetElementsAsFloat(long index, float[] value) => throw new NotSupportedException();
        public override void SetElementsAsHalf(long index, TensorSharp.half[] value) => throw new NotSupportedException();
        public override void CopyToStorage(long storageIndex, IntPtr src, long byteCount) => throw new NotSupportedException();
        public override void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount) => throw new NotSupportedException();
    }
}
