using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public class Qwen35CacheMemoryPolicyTests
{
    [Theory]
    [InlineData(0, 8448)]
    [InlineData(256 * 128 * 2, 8448)]
    [InlineData(8192 * 128 * 2 - 1, 8448)]
    [InlineData(8192 * 128 * 2, 16384)]
    public void GrowthAcross8192_OnlyReservesGeometricSlackWhenAdditionalBytesFit(long spare, int expected)
    {
        // Two F32 attention layers, two KV heads, head dimension four: 128 bytes/token.
        // This is the observed 8370-token boundary; the existing 8192 rows are already resident.
        Assert.Equal(expected, Qwen35Model.ResolveAttentionCacheGrowthCapacity(8192, 8370, 32768, 128, spare));
    }

    [Fact]
    public void GrowthPolicy_PreservesRequiredRowsAndContextLimitWithoutIntegerOverflow()
    {
        Assert.Equal(8192, Qwen35Model.ResolveAttentionCacheGrowthCapacity(8192, 8100, 32768, 128, 0));
        Assert.Equal(16384, Qwen35Model.ResolveAttentionCacheGrowthCapacity(8192, 8370, 32768, 128, null));
        Assert.Equal(8448, Qwen35Model.ResolveAttentionCacheGrowthCapacity(8192, 8370, 32768, 128, long.MaxValue, false));
        Assert.Equal(8371, Qwen35Model.ResolveAttentionCacheGrowthCapacity(8192, 8370, 8371, 128, 0));
        Assert.Equal(int.MaxValue, Qwen35Model.ResolveAttentionCacheGrowthCapacity(1 << 30, int.MaxValue - 1, int.MaxValue, 128, null));
        Assert.Throws<ArgumentOutOfRangeException>(() => Qwen35Model.ResolveAttentionCacheGrowthCapacity(8192, 9000, 8500, 128, 0));
    }

    [Fact]
    public void IdleBudget_IncludesExistingPoolAndHonorsCountLimit()
    {
        Assert.True(Qwen35Model.CanPoolIdleCache(512, 512, 1, 64, 2048));
        Assert.False(Qwen35Model.CanPoolIdleCache(513, 512, 1, 64, 2048));
        Assert.False(Qwen35Model.CanPoolIdleCache(1, 0, 0, 64, 0));
        Assert.False(Qwen35Model.CanPoolIdleCache(1, 0, 0, 0, null));
        Assert.False(Qwen35Model.CanPoolIdleCache(1, 0, 4, 4, long.MaxValue));
        Assert.False(Qwen35Model.CanPoolIdleCache(long.MaxValue, long.MaxValue, 1, 64, long.MaxValue));
        Assert.True(Qwen35Model.CanPoolIdleCache(1, 0, 0, 64, null));
    }

    [Theory]
    [InlineData(0, 8448)]
    [InlineData(long.MaxValue, 16384)]
    public void ActualGrowth_PreservesEveryLiveKvRowAcrossHeadStrides(long spare, int expectedCapacity)
    {
        using var fixture = new Fixture();
        var model = fixture.Model;
        model.SpareBytes = long.MaxValue;
        object parked = fixture.Holder();
        fixture.Retained.Add("optional-idle", parked);
        model.DiscardRetainedCache("optional-idle");
        model.SpareBytes = spare;
        const int capacity = 8192, live = 8000, heads = 2, dim = 4, layers = 2;
        Set(typeof(ModelBase), model, "_cacheSeqLen", live);
        Set(typeof(Qwen35Model), model, "_kvCacheCapacity", capacity);
        var oldK = new Tensor[layers];
        var oldV = new Tensor[layers];
        for (int l = 0; l < layers; l++)
        {
            oldK[l] = fixture.Tensor(heads, capacity, dim);
            oldV[l] = fixture.Tensor(heads, capacity, dim);
            var values = new float[heads * capacity * dim];
            for (int h = 0; h < heads; h++)
                for (int p = 0; p < capacity; p++)
                    for (int d = 0; d < dim; d++)
                        values[(h * capacity + p) * dim + d] = p < live ? Value(l, h, p, d) : -123456;
            oldK[l].SetElementsAsFloat(values);
            oldV[l].SetElementsAsFloat(values.Select(x => -x).ToArray());
        }
        Set(typeof(Qwen35Model), model, "_kvCacheK", oldK.ToArray());
        Set(typeof(Qwen35Model), model, "_kvCacheV", oldV.ToArray());

        Invoke(model, "EnsureCacheCapacity", 8370, true);

        var newK = (Tensor[])Get(typeof(Qwen35Model), model, "_kvCacheK")!;
        var newV = (Tensor[])Get(typeof(Qwen35Model), model, "_kvCacheV")!;
        fixture.Own(newK.Concat(newV));
        Assert.Equal(expectedCapacity, Get(typeof(Qwen35Model), model, "_kvCacheCapacity"));
        AssertHolderDisposed(parked, spare == 0);
        Assert.Equal(live, model.CacheSeqLen);
        for (int l = 0; l < layers; l++)
        {
            Assert.True(IsDisposed(oldK[l]));
            Assert.True(IsDisposed(oldV[l]));
            float[] k = newK[l].GetElementsAsFloat(heads * expectedCapacity * dim);
            float[] v = newV[l].GetElementsAsFloat(heads * expectedCapacity * dim);
            for (int h = 0; h < heads; h++)
                for (int p = 0; p < live; p++)
                    for (int d = 0; d < dim; d++)
                    {
                        int index = (h * expectedCapacity + p) * dim + d;
                        Assert.Equal(Value(l, h, p, d), k[index]);
                        Assert.Equal(-Value(l, h, p, d), v[index]);
                    }
        }
        Invoke(model, "EnsureCacheCapacity", 8371, true);
        Assert.Same(newK[0], ((Tensor[])Get(typeof(Qwen35Model), model, "_kvCacheK")!)[0]);
    }

    [Fact]
    public void IdleReuse_PreservesStorageIdentityAndResetsRecurrentState()
    {
        using var fixture = new Fixture();
        fixture.Model.SpareBytes = long.MaxValue;
        object holder = fixture.Holder();
        var k = ((Tensor[])Get(HolderType, holder, "K")!)[0];
        var delta = ((Tensor[])Get(HolderType, holder, "DeltaState")!)[1];
        fixture.Retained.Add("finished", holder);
        fixture.Model.DiscardRetainedCache("finished");
        Assert.False(fixture.Retained.Contains("finished"));
        Assert.False(IsDisposed(k));

        object reused = Invoke(fixture.Model, "CreateFreshHolder")!;
        Assert.Same(holder, reused);
        Assert.Same(k, ((Tensor[])Get(HolderType, reused, "K")!)[0]);
        Assert.Equal(0, Get(HolderType, reused, "CacheSeqLen"));
        foreach (string flag in new[] { "KvHostDirty", "GdnHostDirty", "FdStateResident", "ArenaStateResident" })
            Assert.False((bool)Get(HolderType, reused, flag)!);
        Assert.All(((float[][])Get(HolderType, reused, "ConvState")!)[1], x => Assert.Equal(0, x));
        Assert.All((int[])Get(HolderType, reused, "ConvWriteIdx")!, x => Assert.Equal(0, x));
        Assert.All(delta.GetElementsAsFloat(8), x => Assert.Equal(0, x));
        // KV is logically empty; masked unwritten rows can retain bytes without an expensive fill.
        Assert.All(k.GetElementsAsFloat(32), x => Assert.Equal(7, x));
    }

    [Fact]
    public void EvictionUnderPressure_FreesIncomingAndParkedHoldersButPreservesLiveAndRetainedState()
    {
        using var fixture = new Fixture();
        fixture.Model.SpareBytes = long.MaxValue;
        object parked = fixture.Holder();
        fixture.Retained.Add("parked", parked);
        fixture.Model.DiscardRetainedCache("parked");
        object incoming = fixture.Holder();
        object live = fixture.Holder();
        object retained = fixture.Holder();
        fixture.Retained.Add("incoming", incoming);
        fixture.Retained.Add("still-retained", retained);
        fixture.Active.Add("live", live);
        fixture.Model.SpareBytes = 0;

        fixture.Model.DiscardRetainedCache("incoming");

        AssertHolderDisposed(parked, true);
        AssertHolderDisposed(incoming, true);
        AssertHolderDisposed(live, false);
        AssertHolderDisposed(retained, false);
        Assert.Same(live, fixture.Active["live"]);
        Assert.Same(retained, fixture.Retained["still-retained"]);
        var pool = (IList?)Get(typeof(Qwen35Model), fixture.Model, "_holderPool");
        Assert.True(pool == null || pool.Count == 0);
    }

    [Fact]
    public void AggregateIdleBytes_TriggerRealDisposalBeforeHolderCountLimit()
    {
        using var fixture = new Fixture();
        // Each fixture owns K/V 2*128, delta 32, conv 12, indices 8, logits 16 = 324 bytes.
        // A 648-byte spare budget permits exactly one holder at the half-spare policy.
        fixture.Model.SpareBytes = 648;
        object first = fixture.Holder(), second = fixture.Holder();
        fixture.Retained.Add("first", first);
        fixture.Retained.Add("second", second);
        fixture.Model.DiscardRetainedCache("first");
        AssertHolderDisposed(first, false);
        fixture.Model.DiscardRetainedCache("second");
        AssertHolderDisposed(first, true);
        AssertHolderDisposed(second, true);
    }

    [Fact]
    public void DisabledPool_FreesReleasedHolderEvenWithAmpleMemory()
    {
        using var fixture = new Fixture();
        Environment.SetEnvironmentVariable("TS_KV_HOLDER_POOL_MAX", "0");
        fixture.Model.SpareBytes = long.MaxValue;
        object holder = fixture.Holder();
        fixture.Active.Add("finished", holder);
        fixture.Model.OnSequenceReleased("finished");
        Assert.False(fixture.Active.Contains("finished"));
        AssertHolderDisposed(holder, true);
    }

    [Fact]
    public void ReleasingActiveGrownHolder_FreesLiveArraysAndRestoresPrimary()
    {
        using var fixture = new Fixture();
        fixture.Model.SpareBytes = 0;
        object stale = fixture.Holder(), live = fixture.Holder(), primary = fixture.Holder();
        fixture.Active.Add("grown", stale);
        Set(typeof(Qwen35Model), fixture.Model, "_activeFusedKey", "grown");
        Set(typeof(Qwen35Model), fixture.Model, "_primaryHolder", primary);
        Set(typeof(Qwen35Model), fixture.Model, "_kvCacheK", Get(HolderType, live, "K")!);
        Set(typeof(Qwen35Model), fixture.Model, "_kvCacheV", Get(HolderType, live, "V")!);
        Set(typeof(Qwen35Model), fixture.Model, "_deltaStateTensor", Get(HolderType, live, "DeltaState")!);
        Set(typeof(Qwen35Model), fixture.Model, "_kvCacheCapacity", 16);
        Set(typeof(ModelBase), fixture.Model, "_cacheSeqLen", 9);

        fixture.Model.OnSequenceReleased("grown");

        Assert.False(fixture.Active.Contains("grown"));
        AssertHolderDisposed(live, true);
        AssertHolderDisposed(stale, false);
        AssertHolderDisposed(primary, false);
        Assert.Same(Get(HolderType, primary, "K"), Get(typeof(Qwen35Model), fixture.Model, "_kvCacheK"));
        Assert.Equal(3, fixture.Model.CacheSeqLen);
        Assert.Null(Get(typeof(Qwen35Model), fixture.Model, "_activeFusedKey"));
    }

    private static float Value(int layer, int head, int position, int channel) => layer * 1000000 + head * 100000 + position * 4 + channel;
    private static readonly Type HolderType = typeof(Qwen35Model).GetNestedType("Qwen35KvCacheHolder", BindingFlags.NonPublic)!;
    private static object? Invoke(object model, string name, params object[] args) => typeof(Qwen35Model).GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(model, args);
    private static object? Get(Type type, object target, string name) => type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)!.GetValue(target);
    private static void Set(Type type, object target, string name, object value) => type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)!.SetValue(target, value);
    private static bool IsDisposed(Tensor tensor) => (int)Get(typeof(Tensor), tensor, "isDisposed")! != 0;
    private static void AssertHolderDisposed(object holder, bool expected)
    {
        foreach (string field in new[] { "K", "V", "DeltaState" })
            foreach (Tensor? tensor in (Tensor[])Get(HolderType, holder, field)!)
                if (tensor != null) Assert.Equal(expected, IsDisposed(tensor));
    }

    private sealed class MemoryTestModel : Qwen35Model
    {
        // Never called: the fixture constructs only cache ownership state, without a GGUF/backend.
        public MemoryTestModel() : base("", BackendType.Cpu) { }
        public long? SpareBytes;
        protected override long? GetCacheMemorySpareBytes() => SpareBytes;
    }

    private sealed class Fixture : IDisposable
    {
        public MemoryTestModel Model { get; } = (MemoryTestModel)RuntimeHelpers.GetUninitializedObject(typeof(MemoryTestModel));
        public IDictionary Retained { get; } = NewDictionary();
        public IDictionary Active { get; } = NewDictionary();
        private readonly CpuAllocator allocator = new(BlasEnum.DotNet);
        private readonly List<Tensor> tensors = new();
        private readonly string? oldPoolMax = Environment.GetEnvironmentVariable("TS_KV_HOLDER_POOL_MAX");
        public Fixture()
        {
            Environment.SetEnvironmentVariable("TS_KV_HOLDER_POOL_MAX", "64");
            Set(typeof(ModelBase), Model, "<Config>k__BackingField", new ModelConfig { NumLayers = 2, NumKVHeads = 2, HiddenSize = 8, NumHeads = 2 });
            Set(typeof(ModelBase), Model, "<ExecutionPlan>k__BackingField", new BackendExecutionPlan(BackendType.Cpu));
            Set(typeof(ModelBase), Model, "_allocator", allocator);
            Set(typeof(ModelBase), Model, "_backend", BackendType.Cpu);
            Set(typeof(ModelBase), Model, "_maxContextLength", 32768);
            Set(typeof(ModelBase), Model, "_kvCacheDtype", KvCacheDtype.F32);
            Set(typeof(Qwen35Model), Model, "_isRecurrent", new[] { false, false });
            Set(typeof(Qwen35Model), Model, "_retainedFusedHolders", Retained);
            Set(typeof(Qwen35Model), Model, "_fusedHolders", Active);
        }
        public Tensor Tensor(params long[] shape)
        {
            var tensor = new Tensor(allocator, DType.Float32, shape);
            tensors.Add(tensor);
            return tensor;
        }
        public void Own(IEnumerable<Tensor> items) => tensors.AddRange(items);
        public object Holder()
        {
            object holder = RuntimeHelpers.GetUninitializedObject(HolderType);
            Tensor k = Tensor(2, 8, 2), v = Tensor(2, 8, 2), delta = Tensor(8);
            k.SetElementsAsFloat(Enumerable.Repeat(7f, 32).ToArray());
            delta.SetElementsAsFloat(Enumerable.Repeat(9f, 8).ToArray());
            Set(HolderType, holder, "K", new Tensor[] { k, null! });
            Set(HolderType, holder, "V", new Tensor[] { v, null! });
            Set(HolderType, holder, "DeltaState", new Tensor[] { null!, delta });
            Set(HolderType, holder, "KvCapacity", 8);
            Set(HolderType, holder, "CacheSeqLen", 3);
            Set(HolderType, holder, "ConvState", new float[][] { null!, new float[] { 1, 2, 3 } });
            Set(HolderType, holder, "ConvWriteIdx", new[] { 0, 1 });
            Set(HolderType, holder, "Logits", new float[4]);
            foreach (string flag in new[] { "KvHostDirty", "GdnHostDirty", "FdStateResident", "ArenaStateResident" })
                Set(HolderType, holder, flag, true);
            return holder;
        }
        public void Dispose()
        {
            foreach (Tensor tensor in tensors) tensor.Dispose();
            Environment.SetEnvironmentVariable("TS_KV_HOLDER_POOL_MAX", oldPoolMax);
        }
        private static IDictionary NewDictionary() => (IDictionary)Activator.CreateInstance(typeof(Dictionary<,>).MakeGenericType(typeof(string), HolderType))!;
    }
}
