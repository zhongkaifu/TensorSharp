// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests;

/// <summary>Complete production holder transfer with injected device I/O and
/// real budgeted host allocations. These CPU fixtures qualify export ordering,
/// metadata and failure ownership; they do not qualify native/GPU continuation.</summary>
[Collection(EngineEnvironmentCollection.Name)]
public sealed class Qwen4ExpHostKvDemotionTests
{
    [Fact]
    public void CompleteStateDemotionPreservesMetadataAndDonationWithoutDuplicatingHolder()
    {
        using var f = new Fixture();
        long hostBefore = f.Budget.Snapshot().Single(p => p.Pool == "ram").Committed;
        Assert.Equal(1, f.Model.UnifiedRetainedDeviceHolderCount);
        Assert.Equal(Environment.SystemPageSize, f.Model.GetUnifiedRetainedHostKvDemotionScratchBytes());
        using var maintenance = f.Maintenance();
        using (f.Scope.EnterExecution(maintenance))
            Assert.Equal(1, f.Model.DemoteUnifiedRetainedHostKv(hooks: f.Hooks()));
        Assert.Equal(1, f.ReleaseCount);
        Assert.Equal(5, f.InvalidationCount);
        Assert.Equal(3, f.ExportCount);
        Assert.True(f.Events.LastIndexOf("export:3") < f.Events.IndexOf("release"));
        Assert.False(f.Get<bool>("KvHostStale"));
        Assert.False(f.Get<bool>("DeviceStateSeeded"));
        Assert.False(f.Get<bool>("DeviceMirrored"));
        Assert.False(f.Get<bool>("Retired"));
        Assert.Equal(1, f.Model.UnifiedRetainedHostKvDemotions);
        Assert.Equal(0, f.Model.UnifiedRetainedDeviceHolderCount);
        Assert.Equal(0, f.DeviceCommitted);
        Assert.Equal(Enumerable.Range(10, 16).Select(i => (float)i), f.Read(f.Conv));
        Assert.Equal(new float[] { 500, 501, 502, 503 }, f.Read(f.State));
        Assert.Equal(Enumerable.Range(100, 8).Select(i => (float)i), f.Read(f.K));
        Assert.Equal(Enumerable.Range(200, 8).Select(i => (float)i), f.Read(f.V));
        Assert.Equal(Enumerable.Range(300, 8).Select(i => (float)i), f.Read(f.Indexer));
        Assert.Equal(new float[] { 90, 91, 92, 93 }, f.Get<float[]>("PleConvState"));
        Assert.Equal(Enumerable.Range(7, 16).Select(i => (float)i), f.Get<float[][]>("GdnConvState")[0]);
        Assert.Equal(new[] { 1, 0 }, f.Get<int[]>("GdnConvWriteIdx"));
        Assert.Equal(Enumerable.Range(1, 12), f.Get<int[]>("QsaPositions"));
        Assert.Equal(3, f.Get<int>("QsaPositionCount"));
        Assert.Equal(new[] { 11, 22 }, f.Get<List<int>>("PleHistory"));
        Assert.Equal(3, f.Get<int>("PleNextPos"));
        Assert.Equal(27, f.Get<int>("MropeCacheGap"));
        Assert.Equal(3, f.Get<int>("CacheSeqLen"));
        Assert.Equal(4, f.Get<int>("KvCapacity"));
        Assert.Equal(new float[] { 1, 2 }, f.Get<float[]>("Logits"));
        // Scratch may remain as a physically charged allocator pool block.
        // Trim demonstrates that exactly the original holder remains in RAM.
        f.Pool.Trim();
        Assert.Equal(hostBefore, f.Budget.Snapshot().Single(p => p.Pool == "ram").Committed);
        Assert.True(f.Model.CanReuseRetainedPrefix("retained", 3, 3));
        Assert.True(f.Model.TryRebindRetainedCache("retained", "continued"));
        Assert.Same(f.Holder, ((IDictionary)Fixture.Field(f.Model, "_fusedHolders").GetValue(f.Model)!)["continued"]);
        Assert.Equal(0, f.Model.DemoteUnifiedRetainedHostKv(hooks: f.Hooks()));
    }

    [Fact]
    public void SuccessfulPhysicalReleaseReportsReducedResidencyForTheSameHostPayload()
    {
        using var f = new Fixture();
        var before = f.Model.MeasureEndState("retained");
        var sink = new ResidencySink(() => Assert.Equal(0, f.DeviceCommitted));
        f.Model.AttachPrefixCache(sink);
        using var maintenance = f.Maintenance();
        using (f.Scope.EnterExecution(maintenance))
            Assert.Equal(1, f.Model.DemoteUnifiedRetainedHostKv(hooks: f.Hooks()));
        var notification = Assert.Single(sink.Demotions);
        Assert.Equal("retained", notification.Key);
        Assert.Equal(before, notification.Previous);
        Assert.Equal(before with { Bytes = before.Bytes with { DeviceKv = 0 } }, notification.Current);
        Assert.Equal(notification.Current, f.Model.MeasureEndState("retained"));
        Assert.True(f.Model.CanReuseRetainedPrefix("retained", 3, 3));
        Assert.Empty(sink.Invalidations);
    }

    private sealed class ResidencySink(Action released) : IPrefixPayloadSink
    {
        internal List<string> Invalidations { get; } = [];
        internal List<(string Key, PayloadFootprint Previous, PayloadFootprint Current)> Demotions { get; } = [];
        public void OnPayloadInvalidated(string key, InvalidationReason reason) => Invalidations.Add(key);
        public void OnPayloadDeviceDemoted(string key, PayloadFootprint previous, PayloadFootprint current)
        { released(); Demotions.Add((key, previous, current)); }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void FailedAuthoritativeExportNeverFreesOrPublishesPartialDeviceState(int failure)
    {
        using var f = new Fixture();
        long before = f.DeviceCommitted;
        using var maintenance = f.Maintenance();
        using (f.Scope.EnterExecution(maintenance))
            Assert.Throws<InvalidOperationException>(() => f.Model.DemoteUnifiedRetainedHostKv(hooks: f.Hooks(failExport: failure)));
        Assert.Equal(0, f.ReleaseCount);
        Assert.Equal(0, f.InvalidationCount);
        Assert.Equal(before, f.DeviceCommitted);
        f.AssertDeviceAuthorityUnpublished();
        f.Pool.Trim();
        Assert.Equal(5, f.Scope.Usage.Allocations);
    }

    [Fact]
    public void CancellationAfterEveryExportKeepsAuthoritativeDeviceOwnerAndCanRetry()
    {
        using var f = new Fixture();
        using var cancelled = new CancellationTokenSource();
        using var maintenance = f.Maintenance();
        using (f.Scope.EnterExecution(maintenance))
            Assert.Throws<OperationCanceledException>(() => f.Model.DemoteUnifiedRetainedHostKv(
                cancellationToken: cancelled.Token, hooks: f.Hooks(afterExports: cancelled.Cancel)));
        Assert.Equal(3, f.ExportCount);
        Assert.Equal(0, f.ReleaseCount);
        Assert.Equal(0, f.InvalidationCount);
        f.AssertDeviceAuthorityUnpublished();
        using (f.Scope.EnterExecution(maintenance))
            Assert.Equal(1, f.Model.DemoteUnifiedRetainedHostKv(hooks: f.Hooks()));
        Assert.Equal(0, f.DeviceCommitted);
    }

    [Fact]
    public void RefusedMaintenanceScratchDoesNotDrawFromClosedRetainedRequestOrEvictDeviceState()
    {
        using var f = new Fixture();
        using var empty = f.Budget.Reserve([new("ram", 0), new("vram", 0)]);
        using (f.Scope.EnterExecution(empty))
            Assert.Throws<MemoryPressureException>(() => f.Model.DemoteUnifiedRetainedHostKv(hooks: f.Hooks()));
        Assert.Equal(0, f.ExportCount);
        Assert.Equal(0, f.ReleaseCount);
        Assert.Equal(5, f.Scope.Usage.Allocations);
        f.AssertDeviceAuthorityUnpublished();
    }

    [Fact]
    public void FlushFailureDoesNotReadStaleSoloStateOrFreeAnyDeviceAllocation()
    {
        using var f = new Fixture();
        using var maintenance = f.Maintenance();
        using (f.Scope.EnterExecution(maintenance))
            Assert.Throws<IOException>(() => f.Model.DemoteUnifiedRetainedHostKv(hooks: f.Hooks(failFlush: true)));
        Assert.Equal(0, f.ExportCount);
        Assert.Equal(0, f.ReleaseCount);
        Assert.Equal(5, f.Scope.Usage.Allocations);
        f.AssertDeviceAuthorityUnpublished();
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FailedPhysicalEvictionKeepsRetryOwnerChargedAndQuarantinesReuse(bool failInvalidation)
    {
        using var f = new Fixture();
        long before = f.DeviceCommitted;
        using var maintenance = f.Maintenance();
        using (f.Scope.EnterExecution(maintenance))
            Assert.Throws<IOException>(() => f.Model.DemoteUnifiedRetainedHostKv(
                hooks: f.Hooks(failRelease: !failInvalidation, failInvalidation: failInvalidation)));
        Assert.Equal(3, f.ExportCount);
        Assert.True(f.Get<bool>("DeviceStateSeeded"));
        Assert.True(f.Get<bool>("DeviceMirrored"));
        Assert.True(f.Get<bool>("SpecStateFailed"));
        Assert.True(f.Get<bool>("Retired"));
        Assert.False(f.Model.CanReuseRetainedPrefix("retained", 3, 3));
        Assert.False(f.Model.TryRebindRetainedCache("retained", "unsafe"));
        Assert.Equal(1, f.Model.RetainedCacheCount);
        Assert.True(f.DeviceCommitted > 0);
        Assert.True(f.DeviceCommitted <= before);
        Assert.Equal(0, f.Model.UnifiedRetainedHostKvDemotions);
    }

    [Theory]
    [InlineData("gdn")]
    [InlineData("qsa")]
    [InlineData("ple")]
    public void IncompleteHybridStateIsRejectedBeforeAnyDeviceTransfer(string section)
    {
        using var f = new Fixture();
        if (section == "gdn") f.Get<Tensor[]>("GdnStateT")[0] = null!;
        if (section == "qsa") f.SetHolder("QsaPositionCount", 4);
        if (section == "ple") f.SetHolder("PleConvState", GC.AllocateArray<float>(3, pinned: true));
        Assert.Throws<InvalidOperationException>(() => f.Model.GetUnifiedRetainedHostKvDemotionScratchBytes());
        using var maintenance = f.Budget.Reserve([new("ram", Environment.SystemPageSize), new("vram", 0)]);
        using (f.Scope.EnterExecution(maintenance))
            Assert.Throws<InvalidOperationException>(() => f.Model.DemoteUnifiedRetainedHostKv(hooks: f.Hooks()));
        Assert.Empty(f.Events);
        f.AssertDeviceAuthorityUnpublished();
    }

    [Fact]
    public void LiveAliasesMtpAndMultiDeviceLayoutsDoNotEnterRetainedDemotionRoute()
    {
        using var f = new Fixture();
        var active = Fixture.Field(f.Model, "_fusedHolders");
        var holders = (IDictionary)Activator.CreateInstance(active.FieldType)!;
        holders["live"] = f.Holder; active.SetValue(f.Model, holders);
        Assert.Equal(0, f.Model.UnifiedRetainedDeviceHolderCount);
        Assert.Equal(0, f.Model.DemoteUnifiedRetainedHostKv(hooks: f.Hooks()));
        holders.Clear();
        f.SetModel("_mtpReady", true);
        Assert.False(f.Model.SupportsUnifiedRetainedHostKvDemotion);
        Assert.Equal(0, f.Model.DemoteUnifiedRetainedHostKv(hooks: f.Hooks()));
        f.SetModel("_mtpReady", false);
        Fixture.SetBase(f.Model, "<LayerSplitDegree>k__BackingField", 2);
        Assert.False(f.Model.SupportsUnifiedRetainedHostKvDemotion);
        Assert.Equal(0, f.Model.DemoteUnifiedRetainedHostKv(hooks: f.Hooks()));
        Assert.Empty(f.Events);
        f.AssertDeviceAuthorityUnpublished();
    }

    private sealed class Fixture : IDisposable
    {
        internal readonly MemoryBudget Budget = new([new("ram", 1L << 20), new("vram", 1L << 20)]);
        internal readonly HostAllocationBudgetScope Scope;
        internal readonly GgmlMemoryPool Pool = new(GgmlBackendType.Cuda);
        internal readonly Qwen4ExpModel Model = (Qwen4ExpModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen4ExpModel));
        internal readonly object Holder;
        internal readonly Tensor Conv, State, K, V, Indexer;
        internal readonly List<string> Events = new();
        internal int ReleaseCount, InvalidationCount, ExportCount;
        private readonly List<Tensor> _tensors = new();
        private readonly Dictionary<Tensor, BudgetReservation> _deviceMirrors = new();
        private readonly Dictionary<IntPtr, byte[]> _stateSources = new();
        private readonly BudgetReservation _deviceState;
        private readonly IAllocator _allocator;

        internal Fixture()
        {
            Scope = new(Budget, ["ram"]);
            var context = (GgmlContext)RuntimeHelpers.GetUninitializedObject(typeof(GgmlContext));
            typeof(GgmlContext).GetField("<MemoryPool>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(context, Pool);
            typeof(GgmlContext).GetField("<BackendType>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(context, GgmlBackendType.Cuda);
            _allocator = new CpuTransferAllocator(context);
            SetBase(Model, "_allocator", _allocator);
            SetBase(Model, "_backend", BackendType.GgmlCuda);
            SetBase(Model, "<ExecutionPlan>k__BackingField", new BackendExecutionPlan(BackendType.GgmlCuda));
            SetBase(Model, "<Config>k__BackingField", new ModelConfig { NumLayers = 2, NumKVHeads = 1, KeyLength = 2, ValueLength = 2 });
            SetModel("_isRecurrent", new[] { true, false });
            SetModel("_headVDim", 2); SetModel("_headKDim", 2); SetModel("_numVHeads", 1);
            SetModel("_convKernel", 3); SetModel("_convDim", 8);
            SetModel("_indexerHeads", 1); SetModel("_indexerHeadDim", 2); SetModel("_compressRatios", new[] { 0, 1 });
            SetModel("_pleHeads", 1); SetModel("_pleConvKernel", 2); SetModel("_pleNgram", 2); SetModel("_hcDim", 2);
            SetModel("_pleLayerIndex", 0);
            SetModel("_kCache", new Tensor[2]); // A distinct, inactive primary.
            using var oldRequest = Budget.Reserve([new("ram", 8L * Environment.SystemPageSize), new("vram", 1024)]);
            using (Scope.EnterExecution(oldRequest))
            {
                Conv = Tensor(2, 8); State = Tensor(1, 2, 2);
                K = Tensor(1, 4, 2); V = Tensor(1, 4, 2); Indexer = Tensor(1, 4, 2);
            }
            foreach (var tensor in _tensors)
            {
                var charge = oldRequest.TryTake([new("ram", 0), new("vram", 64)])!;
                charge.Commit(); _deviceMirrors.Add(tensor, charge);
            }
            _deviceState = oldRequest.TryTake([new("ram", 0), new("vram", 128)])!; _deviceState.Commit();
            var retained = Field(Model, "_retainedFusedHolders");
            var holders = (IDictionary)Activator.CreateInstance(retained.FieldType)!;
            Holder = Activator.CreateInstance(retained.FieldType.GenericTypeArguments[1], nonPublic: true)!;
            holders["retained"] = Holder; retained.SetValue(Model, holders);
            SetHolder("K", new[] { null!, K }); SetHolder("V", new[] { null!, V }); SetHolder("IdxK", new[] { null!, Indexer });
            SetHolder("GdnConvStateT", new[] { Conv, null! }); SetHolder("GdnStateT", new[] { State, null! });
            SetHolder("GdnConvState", new[] { Enumerable.Range(7, 16).Select(i => (float)i).ToArray(), null! });
            SetHolder("GdnConvWriteIdx", new[] { 1, 0 });
            SetHolder("QsaPositions", GC.AllocateArray<int>(12, pinned: true));
            Enumerable.Range(1, 12).ToArray().CopyTo(Get<int[]>("QsaPositions"), 0);
            SetHolder("QsaPositionCount", 3); SetHolder("CacheSeqLen", 3); SetHolder("KvCapacity", 4);
            SetHolder("PleConvState", GC.AllocateArray<float>(4, pinned: true));
            SetHolder("PleHistory", new List<int> { 11, 22 }); SetHolder("PleNextPos", 3); SetHolder("MropeCacheGap", 27);
            SetHolder("Logits", new float[] { 1, 2 }); SetHolder("DeviceStateSeeded", true); SetHolder("DeviceMirrored", true); SetHolder("KvHostStale", true);
            byte[] gdn = new byte[272];
            FloatBytes(Enumerable.Range(10, 16).Select(i => (float)i).ToArray()).CopyTo(gdn, 0);
            FloatBytes([500, 501, 502, 503]).CopyTo(gdn, 256);
            _stateSources.Add(TensorComputePrimitives.GetStoragePointer(Conv), gdn);
            _stateSources.Add(TensorComputePrimitives.GetStoragePointer(Indexer), FloatBytes(Enumerable.Range(300, 8).Select(i => (float)i).ToArray()));
            _stateSources.Add(Marshal.UnsafeAddrOfPinnedArrayElement(Get<float[]>("PleConvState"), 0), FloatBytes([90, 91, 92, 93]));
            // oldRequest closes here: retained physical children keep all real
            // allocations charged, but its unused credit cannot fund scratch.
        }
        internal BudgetReservation Maintenance() => Budget.Reserve([new("ram", Model.GetUnifiedRetainedHostKvDemotionScratchBytes()), new("vram", 0)]);
        internal long DeviceCommitted => Budget.Snapshot().Single(p => p.Pool == "vram").Committed;
        internal Qwen4ExpModel.HostKvTransferHooks Hooks(int failExport = 0, bool failFlush = false,
            bool failRelease = false, bool failInvalidation = false, Action? afterExports = null) => new()
        {
            FlushArena = _ => { Events.Add("flush"); if (failFlush) throw new IOException("flush refused"); },
            SyncTensor = tensor =>
            {
                Events.Add("sync"); int start = ReferenceEquals(tensor, K) ? 100 : 200;
                Write(tensor, Enumerable.Range(start, 8).Select(i => (float)i).ToArray());
            },
            ExportState = (key, target, bytes, rank) =>
            {
                Assert.Equal(0, rank); ExportCount++; Events.Add("export:" + ExportCount);
                if (ExportCount == failExport) return false;
                var source = _stateSources[key]; Assert.Equal(bytes, source.LongLength);
                Marshal.Copy(source, 0, target, source.Length);
                if (ExportCount == 3) afterExports?.Invoke();
                return true;
            },
            ReleaseState = keys =>
            {
                Events.Add("release"); ReleaseCount++; Assert.Equal(4, keys.Length);
                if (failRelease) throw new IOException("native state release refused");
                _deviceState.Dispose();
            },
            InvalidateTensor = tensor =>
            {
                Events.Add("invalidate"); InvalidationCount++;
                if (failInvalidation) throw new IOException("device mirror release refused");
                _deviceMirrors[tensor].Dispose();
            },
        };
        internal void AssertDeviceAuthorityUnpublished()
        {
            Assert.True(Get<bool>("DeviceStateSeeded")); Assert.True(Get<bool>("DeviceMirrored"));
            Assert.True(Get<bool>("KvHostStale")); Assert.Equal(0, Model.UnifiedRetainedHostKvDemotions);
        }
        private Tensor Tensor(params long[] dims)
        {
            var tensor = new Tensor(_allocator, DType.Float32, dims); _tensors.Add(tensor);
            Write(tensor, Enumerable.Repeat(-1f, checked((int)(tensor.Storage.ByteLength / 4))).ToArray());
            return tensor;
        }
        internal float[] Read(Tensor tensor)
        {
            tensor.Storage.EnsureHostReadable(); var result = new float[checked((int)(tensor.Storage.ByteLength / 4))];
            Marshal.Copy(TensorComputePrimitives.GetStoragePointer(tensor), result, 0, result.Length); return result;
        }
        private static void Write(Tensor tensor, float[] values)
        {
            tensor.Storage.EnsureHostReadable(); Marshal.Copy(values, 0, TensorComputePrimitives.GetStoragePointer(tensor), values.Length);
        }
        private static byte[] FloatBytes(float[] values)
        {
            var bytes = new byte[values.Length * 4]; Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length); return bytes;
        }
        internal T Get<T>(string name) => (T)Holder.GetType().GetField(name)!.GetValue(Holder)!;
        internal void SetHolder(string name, object value) => Holder.GetType().GetField(name)!.SetValue(Holder, value);
        internal void SetModel(string name, object value) => Field(Model, name).SetValue(Model, value);
        internal static FieldInfo Field(object instance, string name) => instance.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!;
        internal static void SetBase(object instance, string name, object value) => typeof(ModelBase).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(instance, value);
        // Actual instrumented virtual-memory storage, with a synchronous CPU
        // transfer barrier. No native library is initialized by these fixtures.
        private sealed class CpuTransferAllocator(GgmlContext context) : IAllocator
        {
            private readonly GgmlAllocator _inner = new(context, 0);
            public BlasEnum BlasEnum => BlasEnum.GGML_CPU;
            public int DeviceId => 0;
            public Storage Allocate(DType elementType, long elementCount)
                => new CpuTransferStorage(_inner, context, elementType, elementCount);
            public float GetAllocatedMemoryRatio() => 0;
        }
        private sealed class CpuTransferStorage(GgmlAllocator allocator, GgmlContext context, DType elementType, long elementCount)
            : GgmlStorage(allocator, context, elementType, elementCount)
        {
            public override void EnsureHostReadable() { }
        }
        public void Dispose()
        {
            foreach (var tensor in _tensors) tensor.Dispose();
            Pool.Close();
            _deviceState.Dispose(); foreach (var charge in _deviceMirrors.Values) charge.Dispose();
            Assert.All(Budget.Snapshot(), p => { Assert.Equal(0, p.Committed); Assert.Equal(0, p.Reserved); });
            Scope.Dispose();
        }
    }
}
