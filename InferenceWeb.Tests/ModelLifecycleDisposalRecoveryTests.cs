// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

public sealed class ModelLifecycleDisposalRecoveryTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), $"ts-disposal-retry-{Guid.NewGuid():N}");
    public ModelLifecycleDisposalRecoveryTests() => Directory.CreateDirectory(_directory);
    public void Dispose() { try { Directory.Delete(_directory, true); } catch { } }

    [Fact]
    public void FailedUnloadKeepsItsOwnerAndBlocksReplacementUntilCleanupSucceeds()
    {
        string firstPath = ModelFile("first.gguf"), secondPath = ModelFile("second.gguf");
        var budget = new MemoryBudget([new("ram", 128)]);
        FakeModel? first = null;
        int factoryCalls = 0;
        using var service = new ModelLifecycleService(null, (path, _, _, _) =>
        {
            factoryCalls++;
            if (factoryCalls == 1) return first = new FakeModel(path, failures: 2, budget: budget);
            Assert.True(first!.Released);
            Assert.Equal(0, budget.Snapshot().Single().Committed);
            return new FakeModel(path);
        });
        service.LoadModel(firstPath, null, "cpu");
        long epoch = service.LoadEpoch;
        Assert.Throws<IOException>(() => service.Unload());
        Assert.False(service.IsLoaded);
        Assert.Null(service.Model);
        Assert.Null(service.LoadedModelPath);
        Assert.True(first!.IsRetiring);
        Assert.False(first.TryEnterUse());
        Assert.Equal(128, budget.Snapshot().Single().Committed);
        Assert.Equal(epoch + 1, service.LoadEpoch);

        Assert.Throws<IOException>(() => service.LoadModel(secondPath, null, "cpu"));
        Assert.Equal(1, factoryCalls);
        Assert.Equal(2, first.DisposeAttempts);
        Assert.Equal(128, budget.Snapshot().Single().Committed);

        service.LoadModel(secondPath, null, "cpu");
        Assert.Equal(2, factoryCalls);
        Assert.Equal(3, first.DisposeAttempts);
        Assert.Equal(epoch + 1, service.LoadEpoch); // retries are not another model retirement
        Assert.Equal(secondPath, service.LoadedModelPath);
        Assert.Equal(0, budget.Snapshot().Single().Committed);
    }

    [Fact]
    public void ServiceDisposeRetriesItsUnpublishedOwner()
    {
        FakeModel? model = null;
        var service = new ModelLifecycleService(null, (path, _, _, _) => model = new FakeModel(path, failures: 1));
        service.LoadModel(ModelFile("dispose.gguf"), null, "cpu");
        Assert.Throws<IOException>(() => service.Dispose());
        Assert.Null(service.Model);
        Assert.False(model!.Released);
        service.Dispose();
        Assert.True(model.Released);
        Assert.Equal(2, model.DisposeAttempts);
        service.Dispose();
        Assert.Equal(2, model.DisposeAttempts);
    }

    [Fact]
    public void FreeGpuLockDoesNotPermitDisposalWhileARegisteredUseIsStillActive()
    {
        TimeSpan savedTimeout = ModelLifecycleService.UseDrainTimeout;
        FakeModel? first = null;
        bool useHeld = false;
        int factoryCalls = 0;
        var service = new ModelLifecycleService(null, (path, _, _, _) =>
        {
            factoryCalls++;
            var model = new FakeModel(path);
            first ??= model;
            return model;
        });
        try
        {
            ModelLifecycleService.UseDrainTimeout = TimeSpan.Zero;
            service.LoadModel(ModelFile("busy.gguf"), null, "cpu");
            Assert.True(first!.TryEnterUse());
            useHeld = true;
            // Deliberately do not hold GpuComputeLock: an encoder owns tensors
            // between GPU sections too, and that ownership must outlive timeout.
            string replacement = ModelFile("replacement.gguf");
            Assert.Throws<TimeoutException>(() => service.LoadModel(replacement, null, "cpu"));
            Assert.Null(service.Model);
            Assert.True(first.IsRetiring);
            Assert.Equal(0, first.DisposeAttempts);
            Assert.Equal(1, factoryCalls);
            Assert.Throws<TimeoutException>(() => service.Dispose());
            Assert.Equal(0, first.DisposeAttempts);

            first.ExitUse();
            useHeld = false;
            service.LoadModel(replacement, null, "cpu");
            Assert.True(first.Released);
            Assert.Equal(1, first.DisposeAttempts);
            Assert.Equal(2, factoryCalls);
        }
        finally
        {
            if (useHeld) first!.ExitUse();
            ModelLifecycleService.UseDrainTimeout = savedTimeout;
            service.Dispose();
        }
    }

    [Fact]
    public void FailedPartialLoadDisposalBlocksRollbackAndRetainsTheRejectedModel()
    {
        string firstPath = ModelFile("original.gguf"), rejectedPath = ModelFile("worker.gguf");
        var workerGroup = new WorkerGroup();
        FakeModel? rejected = null;
        int factoryCalls = 0;
        using var service = new ModelLifecycleService(null, (path, _, _, _) =>
        {
            factoryCalls++;
            return path == rejectedPath
                ? rejected = new FakeModel(path, failures: 2, group: workerGroup)
                : new FakeModel(path);
        });
        service.LoadModel(firstPath, null, "cpu");
        // A server refuses a distributed worker after the factory returned it.
        // Its initial cleanup and the cleanup before rollback both fail.
        Assert.Throws<IOException>(() => service.LoadModel(rejectedPath, null, "cpu"));
        Assert.Null(service.Model);
        Assert.Equal(2, factoryCalls);
        Assert.Equal(2, rejected!.DisposeAttempts);
        Assert.False(workerGroup.Released);

        service.LoadModel(firstPath, null, "cpu");
        Assert.True(rejected.Released);
        Assert.True(workerGroup.Released);
        Assert.Equal(3, rejected.DisposeAttempts);
        Assert.Equal(3, factoryCalls);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ModelPoolDisposalReleasesCreditsButClosesOnlyAnOwnedContext(bool ownsContext)
    {
        long page = Environment.SystemPageSize;
        var budget = new MemoryBudget([new("ram", page)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
        var pool = new GgmlMemoryPool(GgmlBackendType.Cuda);
        // CPU-only lifetime fixture: exercise ModelBase's pool ownership branch
        // with a real budgeted pool, without initializing any native GPU backend.
        var context = (GgmlContext)RuntimeHelpers.GetUninitializedObject(typeof(GgmlContext));
        typeof(GgmlContext).GetField("<MemoryPool>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(context, pool);
        using var model = new FakeModel(ModelFile("pool.gguf"));
        typeof(ModelBase).GetField("_ggmlContext", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, context);
        typeof(ModelBase).GetField("_ownsGgmlContext", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, ownsContext);
        try
        {
            var pointer = pool.Allocate(1);
            pool.Free(pointer, 1);
            Assert.Equal(page, scope.Usage.Bytes);
            model.Dispose();
            Assert.Equal(0, scope.Usage.Bytes);

            pointer = pool.Allocate(1);
            pool.Free(pointer, 1);
            // Borrowed context remains a usable retaining pool; the closed owner
            // releases any blocks that arrive after the model went away.
            Assert.Equal(ownsContext ? 0 : page, scope.Usage.Bytes);
        }
        finally { pool.Trim(); }
    }

    private string ModelFile(string name)
    {
        string path = Path.Combine(_directory, name);
        using var writer = new BinaryWriter(File.Create(path));
        writer.Write(0x46554747u); writer.Write(3u);
        writer.Write(0UL); writer.Write(0UL); writer.Write(new byte[8]);
        return path;
    }

    private sealed class FakeModel : ModelBase
    {
        private int _failures;
        private BudgetReservation? _allocation;
        public int DisposeAttempts { get; private set; }
        public bool Released { get; private set; }
        public FakeModel(string path, int failures = 0, MemoryBudget? budget = null, ITensorParallelGroup? group = null)
            : base(path, BackendType.Cpu, tpGroup: group)
        {
            _failures = failures;
            _allocation = budget?.Reserve([new("ram", 128)]);
            _allocation?.Commit();
        }
        protected override float[] ForwardCore(int[] tokens) => [];
        protected override void ResetKVCacheCore() { }
        public override void Dispose()
        {
            if (Released) return;
            DisposeAttempts++;
            if (_failures-- > 0) throw new IOException("Injected physical owner release failure.");
            base.Dispose();
            _allocation?.Dispose();
            _allocation = null;
            Released = true;
        }
    }

    private sealed class WorkerGroup : ITensorParallelGroup
    {
        public bool Released { get; private set; }
        public int Degree => 1;
        public bool IsActive => true;
        public int GlobalDegree => 2;
        public int GlobalRankOffset => 1;
        public int NodeCount => 2;
        public IAllocator GetAllocator(int rank) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() => Released = true;
    }
}
