// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Memory;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

/// <summary>CPU lifecycle/hosting checks. No real unified-model or GPU qualification.</summary>
public sealed class UnifiedMemoryHostingTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-unified-host-" + Guid.NewGuid().ToString("N"));
    public UnifiedMemoryHostingTests() => Directory.CreateDirectory(_directory);
    public void Dispose() { try { Directory.Delete(_directory, true); } catch { } }

    [Fact]
    public void OrdinaryModeRetainsTheOriginalFactoryAndSchedulerConfiguration()
    {
        int unifiedCalls = 0;
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance,
            (path, _, _, _) => new FakeModel(path),
            (_, _, _) => { unifiedCalls++; throw new InvalidOperationException(); });
        lifecycle.LoadModel(ModelFile("ordinary.gguf"), null, "cpu");
        var config = Config(4);
        Assert.Same(config, lifecycle.ConfigureScheduler(lifecycle.Model, config));
        Assert.Equal(0, unifiedCalls);
    }

    [Fact]
    public void UnifiedSessionOwnsProjectorLoadSchedulerAndExactlyOneSuccessfulDispose()
    {
        string path = ModelFile("unified.gguf"), projector = ModelFile("projector.gguf");
        int ordinaryCalls = 0, configureCalls = 0, disposals = 0;
        FakeModel model = null;
        var lifecycle = new ModelLifecycleService(NullLogger.Instance,
            (_, _, _, _) => { ordinaryCalls++; throw new InvalidOperationException(); },
            (actualPath, actualProjector, backend) =>
            {
                Assert.Equal(path, actualPath);
                Assert.Equal(projector, actualProjector);
                Assert.Equal(BackendType.GgmlCuda, backend);
                model = new FakeModel(path);
                return new(model, cfg => { configureCalls++; return Config(3); },
                    () => "fixture shared budget; no GPU qualification", () => { disposals++; model.Dispose(); });
            }) { UnifiedMemoryEnabled = true };
        using var host = new InferenceEngineHost(lifecycle, NullLogger.Instance) { SchedulerConfigOverride = Config(8) };
        try
        {
            lifecycle.LoadModel(path, projector, "ggml_cuda");
            Assert.Equal(projector, lifecycle.LoadedMmProjPath);
            Assert.Same(model, lifecycle.Model);
            var engine = host.TryGetEngine();
            Assert.NotNull(engine);
            Assert.Equal(3, engine.PoolStats.totalBlocks);
            Assert.Same(engine, host.TryGetEngine());
            Assert.Equal(1, configureCalls);
            Assert.Equal(0, ordinaryCalls);
            Assert.Throws<InvalidOperationException>(() => lifecycle.UnifiedMemoryEnabled = false);
            host.Reset();
            lifecycle.Unload();
            Assert.Equal(1, disposals);
            Assert.True(model.Released);
            lifecycle.Dispose();
            Assert.Equal(1, disposals);
        }
        finally { host.Reset(); lifecycle.Dispose(); }
    }

    [Fact]
    public void FailedScopeReleaseRetainsSessionAfterModelReleaseAndBlocksNewFactory()
    {
        var budget = new MemoryBudget([new("ram", 16)]);
        var allocation = budget.Reserve([new("ram", 16)]); allocation.Commit();
        int creates = 0, disposeAttempts = 0;
        FakeModel model = null;
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance,
            (_, _, _, _) => throw new InvalidOperationException("No ordinary fallback"),
            (path, _, _) =>
            {
                creates++;
                if (creates > 1) Assert.Equal(0, budget.Snapshot().Single().Committed);
                var loaded = new FakeModel(path); model ??= loaded;
                return new(loaded, cfg => cfg, () => "fixture", () =>
                {
                    loaded.Dispose();
                    if (creates == 1 && ++disposeAttempts <= 2) throw new IOException("Scope close failed.");
                    allocation.Dispose();
                });
            }) { UnifiedMemoryEnabled = true };
        string path = ModelFile("retry.gguf");
        lifecycle.LoadModel(path, null, "ggml_cuda");
        Assert.Throws<IOException>(() => lifecycle.Unload());
        Assert.True(model.Released);
        Assert.Null(lifecycle.Model);
        Assert.Equal(16, budget.Snapshot().Single().Committed);
        Assert.Throws<InvalidOperationException>(() => lifecycle.UnifiedMemoryEnabled = false);
        Assert.Throws<IOException>(() => lifecycle.LoadModel(path, null, "ggml_cuda"));
        Assert.Equal(1, creates);
        lifecycle.LoadModel(path, null, "ggml_cuda");
        Assert.Equal(2, creates);
        Assert.Equal(3, disposeAttempts);
        Assert.Equal(0, budget.Snapshot().Single().Committed);
    }

    [Fact]
    public void FailedConstructionBeforeModelExistsKeepsTheSessionForRetry()
    {
        int creates = 0, disposeAttempts = 0;
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance,
            (_, _, _, _) => throw new InvalidOperationException("No ordinary fallback"),
            (path, _, _) =>
            {
                if (++creates > 1)
                {
                    var model = new FakeModel(path);
                    return new(model, cfg => cfg, () => "fixture", model.Dispose);
                }
                return new(null, cfg => cfg, () => "no model", () =>
                {
                    if (++disposeAttempts <= 2) throw new IOException("Pending pre-model scope.");
                });
            }) { UnifiedMemoryEnabled = true };
        string path = ModelFile("partial.gguf");
        Assert.Throws<IOException>(() => lifecycle.LoadModel(path, null, "ggml_cuda"));
        Assert.Null(lifecycle.Model);
        Assert.Throws<IOException>(() => lifecycle.LoadModel(path, null, "ggml_cuda"));
        Assert.Equal(1, creates);
        lifecycle.LoadModel(path, null, "ggml_cuda");
        Assert.Equal(2, creates);
        Assert.Equal(3, disposeAttempts);
    }

    [Fact]
    public void LiveModelUseDefersSessionAndBudgetDisposal()
    {
        TimeSpan oldTimeout = ModelLifecycleService.UseDrainTimeout;
        FakeModel model = null;
        int disposals = 0;
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance,
            (_, _, _, _) => throw new InvalidOperationException(), (path, _, _) =>
            {
                model = new FakeModel(path);
                return new(model, cfg => cfg, () => "fixture", () => { disposals++; model.Dispose(); });
            }) { UnifiedMemoryEnabled = true };
        bool useHeld = false;
        try
        {
            ModelLifecycleService.UseDrainTimeout = TimeSpan.Zero;
            lifecycle.LoadModel(ModelFile("active.gguf"), null, "ggml_cuda");
            Assert.True(model.TryEnterUse()); useHeld = true;
            Assert.Throws<TimeoutException>(() => lifecycle.Unload());
            Assert.Null(lifecycle.Model);
            Assert.Equal(0, disposals);
            model.ExitUse(); useHeld = false;
            lifecycle.Unload();
            Assert.Equal(1, disposals);
        }
        finally
        {
            if (useHeld) model.ExitUse();
            ModelLifecycleService.UseDrainTimeout = oldTimeout;
        }
    }

    [Fact]
    public void UnifiedStartupDefersUnadmittedKernelWarmup()
    {
        using var model = new FakeModel(ModelFile("warmup-budget.gguf"));
        StartupModelLoader.WarmUpLoadedModel(model, unifiedMemoryEnabled: true, NullLogger.Instance);
        Assert.Equal(0, model.KernelWarmups);
        // The ordinary path must preserve the existing startup behaviour.
        StartupModelLoader.WarmUpLoadedModel(model, unifiedMemoryEnabled: false, NullLogger.Instance);
        Assert.Equal(1, model.KernelWarmups);
    }

    [Fact]
    public void EngineDisposeFailureRemainsOwnedUntilARepeatedDisposeSucceeds()
    {
        FakeModel model = null;
        int sessionDisposals = 0;
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance,
            (_, _, _, _) => throw new InvalidOperationException(), (path, _, _) =>
            {
                model = new FakeModel(path);
                return new(model, cfg => cfg, () => "fixture", () => { sessionDisposals++; model.Dispose(); });
            }) { UnifiedMemoryEnabled = true };
        using var host = new InferenceEngineHost(lifecycle, NullLogger.Instance) { SchedulerConfigOverride = Config(8) };
        lifecycle.LoadModel(ModelFile("engine-release.gguf"), null, "ggml_cuda");
        Assert.NotNull(host.TryGetEngine());
        model.ResetFailure = new IOException("Retained state could not be released.");
        try
        {
            Assert.Throws<IOException>(() => host.Dispose());
            Assert.Null(host.TryGetEngine());
            Assert.Throws<IOException>(() => host.Dispose());
            Assert.Equal(2, model.FailedResets);
            Assert.False(model.Released);
            Assert.Equal(0, sessionDisposals);
        }
        finally { model.ResetFailure = null; host.Dispose(); }
        lifecycle.Unload();
        Assert.True(model.Released);
        Assert.Equal(1, sessionDisposals);
    }

    [Fact]
    public void UnifiedModeRejectsRuntimeSpeculationBeforeRememberingIt()
    {
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance,
            (_, _, _, _) => throw new InvalidOperationException()) { UnifiedMemoryEnabled = true };
        using var host = new InferenceEngineHost(lifecycle, NullLogger.Instance);
        Assert.Throws<NotSupportedException>(() => host.UpdateSpeculation(new SpeculationOptions { Enabled = true }));
        Assert.Null(host.PendingSpeculation);
        Assert.False(host.UpdateSpeculation(SpeculationOptions.Disabled));
    }

    [Fact]
    public void UnifiedModeRefusesAnUnscheduledModelInsteadOfFallingBack()
    {
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance,
            (_, _, _, _) => throw new InvalidOperationException(), (path, _, _) =>
            {
                var model = new FakeModel(path, snapshots: false);
                return new(model, cfg => cfg, () => "fixture", model.Dispose);
            }) { UnifiedMemoryEnabled = true };
        using var host = new InferenceEngineHost(lifecycle, NullLogger.Instance);
        lifecycle.LoadModel(ModelFile("unscheduled.gguf"), null, "ggml_cuda");
        Assert.Throws<NotSupportedException>(() => host.TryGetEngine());
    }

    private static SchedulerConfig Config(int blocks) => new()
    {
        NumBlocks = blocks, BlockSize = 16, MaxNumBatchedTokens = 32,
        MaxPrefillChunkSize = 16, MaxNumRunningSequences = 1, EnablePrefixCaching = false,
    };

    private string ModelFile(string name)
    {
        string path = Path.Combine(_directory, name);
        using var writer = new BinaryWriter(File.Create(path));
        writer.Write(0x46554747u); writer.Write(3u); writer.Write(0UL); writer.Write(0UL); writer.Write(new byte[8]);
        return path;
    }

    private sealed class FakeModel : ModelBase
    {
        private readonly bool _snapshots;
        internal FakeModel(string path, bool snapshots = true) : base(path, BackendType.Cpu)
        {
            _snapshots = snapshots;
            Config = new ModelConfig { Architecture = "unified-host-fixture", NumLayers = 1, VocabSize = 8 };
        }
        public bool Released { get; private set; }
        public Exception ResetFailure { get; set; }
        public int FailedResets { get; private set; }
        public int KernelWarmups { get; private set; }
        public override void WarmUpKernels() => KernelWarmups++;
        public override bool SupportsKVStateSnapshot => _snapshots;
        public override string KVStateFingerprint => "unified-host-fixture";
        public override long ComputeKVBlockByteSize(int tokenCount) => 4L * tokenCount;
        protected override float[] ForwardCore(int[] tokens) => new float[8];
        protected override void ResetKVCacheCore()
        {
            if (ResetFailure is { } failure) { FailedResets++; throw failure; }
        }
        public override void Dispose()
        {
            if (Released) return;
            base.Dispose(); Released = true;
        }
    }
}
