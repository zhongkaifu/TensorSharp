// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>Actual model growth entry points and charged native host allocations.
/// Faults run before backend work; this does not qualify CUDA copy arithmetic.</summary>
[Collection(EngineEnvironmentCollection.Name)]
public sealed class Qwen4ExpCacheGrowthOwnershipTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void InitializationFailureReleasesUnpublishedGrowthAndPreservesOriginal(bool qsa)
    {
        using var f = new Fixture(qsa);
        var failure = new IOException("initialization failed");
        f.Model.CacheGrowthFaultInjector = stage => { if (stage == "allocated") throw failure; };
        Assert.Same(failure, f.Grow());
        Assert.Null(f.Pending);
        Assert.Same(f.Original, f.Published);
        f.Trim();
        Assert.Equal(1, f.Scope.Usage.Allocations);
        f.Model.CacheGrowthFaultInjector = null;
        f.Model.Dispose();
        f.AssertReleased();
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FailedRollbackKeepsOwnerAndOriginalFailureUntilSuccessfulRetry(bool qsa)
    {
        using var f = new Fixture(qsa);
        var creation = new IOException("initialization failed");
        var cleanup = new IOException("invalidation failed");
        f.Model.CacheGrowthFaultInjector = stage => { throw stage == "allocated" ? creation : cleanup; };
        var failure = Assert.IsType<AggregateException>(f.Grow());
        Assert.Same(creation, failure.InnerExceptions[0]);
        Assert.Same(cleanup, failure.InnerExceptions[1]);
        Assert.NotNull(f.Pending);
        Assert.NotSame(f.Original, f.Pending);
        Assert.Same(f.Original, f.Published);
        Assert.Equal(2, f.Scope.Usage.Allocations);
        Tensor pending = f.Pending!;
        // A subsequent attempt must retry cleanup BEFORE allocating a third
        // buffer or replacing the only reference to the failed candidate.
        Assert.Same(cleanup, f.Grow());
        Assert.Same(pending, f.Pending);
        Assert.Equal(2, f.Scope.Usage.Allocations);
        Assert.Same(cleanup, Assert.Throws<IOException>(() => f.Model.Dispose()));
        Assert.Same(pending, f.Pending);
        f.Model.CacheGrowthFaultInjector = null;
        f.Model.Dispose();
        Assert.Null(f.Pending);
        f.AssertReleased();
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PhysicalFreeRefusalRemainsOwnedByPoolUntilDisposalRetry(bool qsa)
    {
        using var f = new Fixture(qsa);
        var failure = new IOException("initialization failed");
        f.Model.CacheGrowthFaultInjector = stage => { if (stage == "allocated") throw failure; };
        f.RefusePhysicalFree = true;
        Assert.Same(failure, f.Grow());
        // Tensor disposal returns storage to the pool; an OS release refusal
        // stays in the pool's pending-release owners with its original credit.
        Assert.Null(f.Pending);
        Assert.Throws<IOException>(() => f.Trim());
        Assert.Equal(2, f.Scope.Usage.Allocations);
        Assert.Throws<InvalidOperationException>(() => f.Scope.Dispose());
        f.Model.CacheGrowthFaultInjector = null;
        Assert.Throws<IOException>(() => f.Model.Dispose());
        Assert.Equal(2, f.Scope.Usage.Allocations);
        f.RefusePhysicalFree = false;
        f.Model.Dispose();
        f.AssertReleased();
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AllocationRefusalDoesNotPublishAnOwnerOrReplaceLiveCache(bool qsa)
    {
        using var f = new Fixture(qsa);
        using var occupied = f.Budget.TryReserve([new("ram", f.Budget.Snapshot().Single().Available)])!;
        Assert.NotNull(occupied);
        Assert.IsType<MemoryPressureException>(f.Grow());
        Assert.Null(f.Pending);
        Assert.Same(f.Original, f.Published);
        Assert.Equal(1, f.Scope.Usage.Allocations);
        occupied.Dispose();
        f.Model.Dispose();
        f.AssertReleased();
    }

    private sealed class Fixture : IDisposable
    {
        internal readonly MemoryBudget Budget = new([new("ram", 8L << 20)]);
        internal readonly HostAllocationBudgetScope Scope;
        internal readonly Qwen4ExpModel Model = (Qwen4ExpModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen4ExpModel));
        internal readonly Tensor Original;
        private readonly GgmlMemoryPool _pool;
        private readonly bool _qsa;
        internal bool RefusePhysicalFree;

        internal Fixture(bool qsa)
        {
            _qsa = qsa;
            Scope = new(Budget, ["ram"]);
            var release = typeof(GgmlMemoryPool).GetMethod("FreeVirtual", BindingFlags.NonPublic | BindingFlags.Static)!;
            _pool = new(GgmlBackendType.Cuda, (pointer, bytes) => !RefusePhysicalFree && (bool)release.Invoke(null, [pointer, bytes])!);
            var context = (GgmlContext)RuntimeHelpers.GetUninitializedObject(typeof(GgmlContext));
            SetField(typeof(GgmlContext), context, "<MemoryPool>k__BackingField", _pool);
            SetField(typeof(GgmlContext), context, "<BackendType>k__BackingField", GgmlBackendType.Cuda);
            var allocator = new GgmlAllocator(context, 0);
            SetField(typeof(ModelBase), Model, "_backend", BackendType.Cpu);
            SetField(typeof(ModelBase), Model, "<ExecutionPlan>k__BackingField", new BackendExecutionPlan(BackendType.Cpu));
            SetField(typeof(ModelBase), Model, "_allocator", allocator);
            SetField(typeof(ModelBase), Model, "_ggmlContext", context);
            SetField(typeof(ModelBase), Model, "_ownsGgmlContext", true);
            foreach (string name in new[] { "_weights", "_quantWeights", "_stackedExpertWeights", "_tpQuantWeights", "_tpWeights" })
            {
                var field = typeof(ModelBase).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!;
                field.SetValue(Model, Activator.CreateInstance(field.FieldType));
            }
            Set("_visionEmbeddingsList", new List<(Tensor Embeddings, int StartPosition)>());
            Set("_usedSlotBases", new HashSet<int> { 0 });
            Set("_indexerHeadDim", 2);
            Original = new Tensor(allocator, DType.Float32, 1, 4, 2);
            Set(qsa ? "_idxKCache" : "_kCache", new[] { Original });
        }

        internal Tensor? Pending => (Tensor?)Field("_pendingCacheGrowth").GetValue(Model);
        internal Tensor? Published => ((Tensor[])Field(_qsa ? "_idxKCache" : "_kCache").GetValue(Model)!)[0];
        internal Exception Grow()
        {
            MethodInfo method = typeof(Qwen4ExpModel).GetMethod(_qsa ? "GrowQsaCache" : "GrowCache", BindingFlags.NonPublic | BindingFlags.Instance)!;
            object[] args = _qsa ? [0, 8, DType.Float32] : [Original, 1, 8, 2, DType.Float32];
            return Assert.Throws<TargetInvocationException>(() => method.Invoke(Model, args)).InnerException!;
        }
        internal long Trim() => _pool.Trim();
        private static FieldInfo Field(string name) => typeof(Qwen4ExpModel).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!;
        private void Set(string name, object value) => Field(name).SetValue(Model, value);
        private static void SetField(Type type, object target, string name, object value)
            => type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)!.SetValue(target, value);
        internal void AssertReleased()
        {
            Assert.Equal(0, Scope.Usage.Allocations); Assert.Equal(0, Scope.Usage.Bytes);
            Assert.Equal(0, Budget.Snapshot().Single().Committed);
            Assert.Equal(0, Budget.Snapshot().Single().Reserved);
        }
        public void Dispose()
        {
            RefusePhysicalFree = false;
            Model.CacheGrowthFaultInjector = null;
            Model.Dispose();
            _pool.Close();
            Scope.Dispose();
        }
    }
}
