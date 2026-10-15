// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>Actual model disposal and physical GGML host-pool ownership, without
/// loading a model file or initializing a native backend.</summary>
[Collection(EngineEnvironmentCollection.Name)]
public sealed class Qwen4ExpDisposalOwnershipTests
{
    [Fact]
    public void ModelDisposalReleasesAll72ActiveRecurrentSeedAllocations()
    {
        using var f = new Fixture();
        f.Set("_gdnConvStateT", Enumerable.Range(0, 36).Select(_ => f.Tensor()).ToArray());
        f.Set("_gdnStateT", Enumerable.Range(0, 36).Select(_ => f.Tensor()).ToArray());
        Assert.Equal(72, f.Scope.Usage.Allocations);
        f.Model.Dispose();
        f.AssertReleased();
        f.Model.Dispose(); // Repeated cleanup must not revisit freed storage.
        f.AssertReleased();
    }

    [Fact]
    public void ModelDisposalReleasesPortableScratchAndUnconsumedVisionEmbeddings()
    {
        using var f = new Fixture();
        foreach (string name in new[] { "_gdnQBuf", "_gdnKBuf", "_gdnVBuf", "_gdnZBuf",
            "_gdnAlphaBuf", "_gdnBetaBuf", "_moeScratchGate", "_moeScratchUp", "_moeScratchDown" })
            f.Set(name, f.Tensor());
        f.Model.SetVisionEmbeddings(f.Tensor(), 0);
        f.Model.SetVisionEmbeddings(f.Tensor(), 128);
        Assert.Equal(11, f.Scope.Usage.Allocations);
        f.Model.Dispose();
        f.AssertReleased();
        f.Model.Dispose();
        f.AssertReleased();
    }

    [Fact]
    public void FailedPhysicalPoolReleaseRetainsCreditUntilModelDisposalRetry()
    {
        using var f = new Fixture();
        f.Set("_gdnConvStateT", new[] { f.Tensor() });
        f.Set("_gdnStateT", new[] { f.Tensor() });
        f.RefusePhysicalFree = true;
        Assert.Throws<IOException>(() => f.Model.Dispose());
        Assert.Equal(2, f.Scope.Usage.Allocations);
        Assert.Throws<InvalidOperationException>(() => f.Scope.Dispose());
        f.RefusePhysicalFree = false;
        f.Model.Dispose();
        f.AssertReleased();
    }

    [Fact]
    public void ModelDisposalReleasesUnpublishedCopyRetryOwner()
    {
        using var f = new Fixture();
        var holderType = typeof(Qwen4ExpModel).GetNestedType("Qwen4ExpKvCacheHolder", BindingFlags.NonPublic)!;
        var holder = Activator.CreateInstance(holderType, nonPublic: true)!;
        holderType.GetField("K")!.SetValue(holder, new[] { f.Tensor() });
        // Represents a failed copy/publication whose cleanup owner was retained.
        f.Set("_pendingHolderAllocation", holder);
        Assert.Same(holder, typeof(Qwen4ExpModel).GetField("_pendingHolderAllocation", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(f.Model));
        Assert.Equal(1, f.Scope.Usage.Allocations);
        f.Model.Dispose();
        f.AssertReleased();
    }

    private sealed class Fixture : IDisposable
    {
        private readonly MemoryBudget _budget = new([new("ram", 8L << 20)]);
        internal readonly HostAllocationBudgetScope Scope;
        internal readonly Qwen4ExpModel Model = (Qwen4ExpModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen4ExpModel));
        private readonly GgmlMemoryPool _pool;
        private readonly GgmlAllocator _allocator;
        private readonly List<Tensor> _tensors = new();
        internal bool RefusePhysicalFree;

        internal Fixture()
        {
            Scope = new(_budget, ["ram"]);
            var release = typeof(GgmlMemoryPool).GetMethod("FreeVirtual", BindingFlags.NonPublic | BindingFlags.Static)!;
            _pool = new(GgmlBackendType.Cuda, (pointer, bytes) => !RefusePhysicalFree && (bool)release.Invoke(null, [pointer, bytes])!);
            var context = (GgmlContext)RuntimeHelpers.GetUninitializedObject(typeof(GgmlContext));
            SetField(typeof(GgmlContext), context, "<MemoryPool>k__BackingField", _pool);
            SetField(typeof(GgmlContext), context, "<BackendType>k__BackingField", GgmlBackendType.Cuda);
            _allocator = new(context, 0);
            // CPU model identity suppresses native graph/cache hooks; storage
            // still uses the real instrumented GGML virtual-memory pool.
            SetField(typeof(ModelBase), Model, "_backend", BackendType.Cpu);
            SetField(typeof(ModelBase), Model, "<ExecutionPlan>k__BackingField", new BackendExecutionPlan(BackendType.Cpu));
            SetField(typeof(ModelBase), Model, "_allocator", _allocator);
            SetField(typeof(ModelBase), Model, "_ggmlContext", context);
            SetField(typeof(ModelBase), Model, "_ownsGgmlContext", true);
            foreach (string name in new[] { "_weights", "_quantWeights", "_stackedExpertWeights", "_tpQuantWeights", "_tpWeights" })
            {
                var field = typeof(ModelBase).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!;
                field.SetValue(Model, Activator.CreateInstance(field.FieldType));
            }
            Set("_visionEmbeddingsList", new List<(Tensor Embeddings, int StartPosition)>());
            Set("_usedSlotBases", new HashSet<int> { 0 });
        }

        internal Tensor Tensor()
        {
            var tensor = new Tensor(_allocator, DType.Float32, 8);
            _tensors.Add(tensor);
            return tensor;
        }
        internal void Set(string name, object value) => SetField(typeof(Qwen4ExpModel), Model, name, value);
        private static void SetField(Type type, object target, string name, object value)
            => type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)!.SetValue(target, value);
        internal void AssertReleased()
        {
            Assert.Equal(0, Scope.Usage.Allocations); Assert.Equal(0, Scope.Usage.Bytes);
            Assert.Equal(0, _budget.Snapshot().Single().Committed);
            Assert.Equal(0, _budget.Snapshot().Single().Reserved);
            Scope.Dispose();
        }
        public void Dispose()
        {
            RefusePhysicalFree = false;
            foreach (var tensor in _tensors) tensor.Dispose();
            _pool.Close();
            Scope.Dispose();
        }
    }
}
