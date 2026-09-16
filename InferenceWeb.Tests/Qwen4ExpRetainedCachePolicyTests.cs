// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>The qwen4exp retention budget and its eviction order, on synthetic
/// holders without a GGUF or a native library. Exactness of the retained /
/// cloned state itself is established on the fixture (Qwen4ExpRetainedCacheTests).</summary>
public sealed class Qwen4ExpRetainedCachePolicyTests
{
    [Fact]
    public void Budget_IncludesRetainedBytes_ClampsToHalfSpare_AndZeroDeclines()
    {
        Assert.True(Qwen4ExpModel.CanRetainWithinBudget(512, 512, 1024, null));
        Assert.False(Qwen4ExpModel.CanRetainWithinBudget(513, 512, 1024, null));
        Assert.False(Qwen4ExpModel.CanRetainWithinBudget(1, 0, 0, null));
        Assert.False(Qwen4ExpModel.CanRetainWithinBudget(-1, 0, 1024, null));
        // Half of the measured headroom caps the configured budget, never raises it.
        Assert.True(Qwen4ExpModel.CanRetainWithinBudget(512, 0, 1024, 1024));
        Assert.False(Qwen4ExpModel.CanRetainWithinBudget(513, 0, 1024, 1024));
        Assert.False(Qwen4ExpModel.CanRetainWithinBudget(1, 0, 1024, 0));
        Assert.True(Qwen4ExpModel.CanRetainWithinBudget(1024, 0, 1024, long.MaxValue));
        Assert.False(Qwen4ExpModel.CanRetainWithinBudget(long.MaxValue, long.MaxValue, long.MaxValue, null));
    }

    [Fact]
    public void Qwen4Exp_AdvertisesExactPrefixReuseOnly()
    {
        // The scheduler's rewind paths key off these: a GDN model must refuse
        // every partial match, and it must say so through the exact-reuse contract.
        Assert.True(typeof(IExactFusedCacheReuse).IsAssignableFrom(typeof(Qwen4ExpModel)));
        Assert.True(typeof(IBatchedPagedModel).IsAssignableFrom(typeof(Qwen4ExpModel)));
        var model = (Qwen4ExpModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen4ExpModel));
        Assert.False(model.SupportsKVCacheTruncation);
    }

    [Fact]
    public void Eviction_DropsOldestConversationsFirst_NeverACheckpoint_AndDeclinesWhenNothingFits()
    {
        using var fixture = new Fixture(budgetBytes: 1000, spareBytes: long.MaxValue);
        object older = fixture.Retain("older", bytes: 400, serial: 1);
        object checkpoint = fixture.Retain("checkpoint", bytes: 300, serial: 2, isCheckpoint: true);
        object newer = fixture.Retain("newer", bytes: 200, serial: 3);

        // 900 retained; 150 more needs one eviction: the OLDEST conversation goes,
        // not the checkpoint that is older than "newer".
        Assert.True(fixture.EnsureBudget(150));
        Assert.False(fixture.Retained.Contains("older"));
        Assert.True(fixture.Retained.Contains("checkpoint"));
        Assert.True(fixture.Retained.Contains("newer"));
        Fixture.AssertDisposed(older, true);
        Fixture.AssertDisposed(checkpoint, false);
        Fixture.AssertDisposed(newer, false);

        // 500 retained now; 600 more evicts "newer" too (300 left) and fits.
        Assert.True(fixture.EnsureBudget(600));
        Assert.False(fixture.Retained.Contains("newer"));
        Assert.True(fixture.Retained.Contains("checkpoint"));
        Fixture.AssertDisposed(newer, true);

        // Nothing but the checkpoint remains and it is never evicted: a holder that
        // cannot fit beside it is declined, and the checkpoint stays intact.
        Assert.False(fixture.EnsureBudget(701));
        Assert.True(fixture.Retained.Contains("checkpoint"));
        Fixture.AssertDisposed(checkpoint, false);
        Assert.True(fixture.EnsureBudget(700));

        // A holder that could not fit even with every conversation gone evicts
        // nothing on its way to being declined.
        object survivor = fixture.Retain("survivor", bytes: 100, serial: 4);
        Assert.False(fixture.EnsureBudget(701));
        Fixture.AssertDisposed(survivor, false);
        Assert.True(fixture.Retained.Contains("survivor"));
    }

    [Fact]
    public void HalfSpareClamp_EvictsEvenUnderAGenerousConfiguredBudget()
    {
        using var fixture = new Fixture(budgetBytes: long.MaxValue / 4, spareBytes: 2000);
        object first = fixture.Retain("first", bytes: 600, serial: 1);
        object second = fixture.Retain("second", bytes: 300, serial: 2);
        // Limit is spare/2 = 1000: 900 retained, 200 more evicts "first".
        Assert.True(fixture.EnsureBudget(200));
        Fixture.AssertDisposed(first, true);
        Fixture.AssertDisposed(second, false);
        Assert.Equal(1, fixture.Retained.Count);
    }

    [Fact]
    public void ZeroBudget_DeclinesWithoutTouchingRetainedHolders()
    {
        using var fixture = new Fixture(budgetBytes: 0, spareBytes: long.MaxValue);
        object kept = fixture.Retain("kept", bytes: 1, serial: 1);
        Assert.False(fixture.EnsureBudget(1));
        Fixture.AssertDisposed(kept, false);
        Assert.True(fixture.Retained.Contains("kept"));
    }

    [Fact]
    public void DiscardAndTrim_FreeConversationsAndKeepCheckpoints()
    {
        using var fixture = new Fixture(budgetBytes: long.MaxValue / 4, spareBytes: null);
        object a = fixture.Retain("a", bytes: 10, serial: 1);
        object ckpt = fixture.Retain("ckpt", bytes: 10, serial: 2, isCheckpoint: true);
        object b = fixture.Retain("b", bytes: 10, serial: 3);
        fixture.Model.DiscardRetainedCache("missing");
        fixture.Model.DiscardRetainedCache("a");
        Fixture.AssertDisposed(a, true);
        Assert.False(fixture.Model.CanReuseRetainedPrefix("a", 3, 3));
        fixture.Model.TrimIdleMemory();
        Fixture.AssertDisposed(b, true);
        Fixture.AssertDisposed(ckpt, false);
        Assert.True(fixture.Retained.Contains("ckpt"));
        Assert.Equal(1, fixture.Retained.Count);
    }

    private sealed class MemoryTestModel : Qwen4ExpModel
    {
        // Never called: the fixture constructs cache ownership state only.
        public MemoryTestModel() : base("", BackendType.Cpu) { }
        public long? SpareBytes;
        protected override long? GetCacheMemorySpareBytes() => SpareBytes;
    }

    private sealed class Fixture : IDisposable
    {
        private static readonly Type HolderType = typeof(Qwen4ExpModel).GetNestedType("Qwen4ExpKvCacheHolder", BindingFlags.NonPublic)!;
        public MemoryTestModel Model { get; } = (MemoryTestModel)RuntimeHelpers.GetUninitializedObject(typeof(MemoryTestModel));
        public IDictionary Retained { get; }
        private readonly CpuAllocator _allocator = new(BlasEnum.DotNet);
        private readonly List<Tensor> _tensors = new();

        public Fixture(long budgetBytes, long? spareBytes)
        {
            Model.SpareBytes = spareBytes;
            Set(typeof(ModelBase), Model, "<Config>k__BackingField", new ModelConfig { NumLayers = 2, NumKVHeads = 1, HiddenSize = 8, NumHeads = 1 });
            Set(typeof(ModelBase), Model, "<ExecutionPlan>k__BackingField", new BackendExecutionPlan(BackendType.Cpu));
            Set(typeof(ModelBase), Model, "_allocator", _allocator);
            Set(typeof(ModelBase), Model, "_backend", BackendType.Cpu);
            Set(typeof(Qwen4ExpModel), Model, "_retainedCacheEnabled", true);
            Set(typeof(Qwen4ExpModel), Model, "_retainedCacheBudgetBytes", budgetBytes);
            Set(typeof(Qwen4ExpModel), Model, "_usedSlotBases", new HashSet<int> { 0 });
            Retained = (IDictionary)Activator.CreateInstance(typeof(Dictionary<,>).MakeGenericType(typeof(string), HolderType))!;
            Set(typeof(Qwen4ExpModel), Model, "_retainedFusedHolders", Retained);
        }

        public object Retain(string key, long bytes, long serial, bool isCheckpoint = false)
        {
            object holder = RuntimeHelpers.GetUninitializedObject(HolderType);
            Tensor k = NewTensor(1, 4, 2), v = NewTensor(1, 4, 2), conv = NewTensor(2, 3), ssm = NewTensor(1, 2, 2);
            Set(HolderType, holder, "K", new Tensor[] { k, null! });
            Set(HolderType, holder, "V", new Tensor[] { v, null! });
            Set(HolderType, holder, "IdxK", new Tensor[] { null!, null! });
            Set(HolderType, holder, "GdnConvStateT", new Tensor[] { null!, conv });
            Set(HolderType, holder, "GdnStateT", new Tensor[] { null!, ssm });
            Set(HolderType, holder, "CacheSeqLen", 3);
            Set(HolderType, holder, "KvCapacity", 4);
            Set(HolderType, holder, "RetainedBytes", bytes);
            Set(HolderType, holder, "RetainedSerial", serial);
            Set(HolderType, holder, "IsCheckpoint", isCheckpoint);
            Set(HolderType, holder, "PleHistory", new List<int>());
            Retained.Add(key, holder);
            return holder;
        }

        public bool EnsureBudget(long bytes)
            => (bool)typeof(Qwen4ExpModel).GetMethod("EnsureRetentionBudget", BindingFlags.Instance | BindingFlags.NonPublic)!
                .Invoke(Model, new object[] { bytes, "test" })!;

        public static void AssertDisposed(object holder, bool expected)
        {
            Assert.Equal(expected, (bool)Get(HolderType, holder, "Disposed")!);
            foreach (string field in new[] { "K", "V", "GdnConvStateT", "GdnStateT" })
                foreach (Tensor? tensor in (Tensor[])Get(HolderType, holder, field)!)
                    if (tensor != null) Assert.Equal(expected, IsDisposed(tensor));
        }

        private Tensor NewTensor(params long[] shape)
        {
            var tensor = new Tensor(_allocator, DType.Float32, shape);
            _tensors.Add(tensor);
            return tensor;
        }

        private static object? Get(Type type, object target, string name)
            => type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)!.GetValue(target);
        private static void Set(Type type, object target, string name, object value)
            => type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)!.SetValue(target, value);
        private static bool IsDisposed(Tensor tensor) => (int)Get(typeof(Tensor), tensor, "isDisposed")! != 0;

        public void Dispose()
        {
            foreach (Tensor tensor in _tensors)
                if (!IsDisposed(tensor)) tensor.Dispose();
        }
    }
}
