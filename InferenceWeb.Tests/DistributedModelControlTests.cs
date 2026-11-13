// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;

namespace InferenceWeb.Tests;

/// <summary>Tests the ModelBase driver/worker route used by distributed
/// whole-model native executors without requiring a GPU or checkpoint.</summary>
public sealed class DistributedModelControlTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"distributed-controls-{Guid.NewGuid():N}.gguf");
    public DistributedModelControlTests() => BackendFailureWarmupTests.WriteProbeGguf(_path);
    public void Dispose() => File.Delete(_path);

    [Fact]
    public void DriverAndWorkerMirrorForwardRefillResetAndTruncate()
    {
        var driverGroup = new ControlGroup(0);
        using var driver = new ControlModel(_path, driverGroup);
        driver.BeginDistributedDriver();
        driver.Forward([1017, 7]);
        driver.ForwardRefill([9]);
        driver.ResetKVCache();
        driver.TruncateKVCache(4);
        Assert.Equal(new[] { 1, 2, 3, 5 }, driverGroup.Messages.Select(message => message.op));

        var workerGroup = new ControlGroup(1);
        foreach (var message in driverGroup.Messages) workerGroup.Messages.Enqueue(message);
        workerGroup.Messages.Enqueue((4, []));
        using var worker = new ControlModel(_path, workerGroup);
        worker.RunDistributedWorkerLoop();
        Assert.Equal(driver.Executed, worker.Executed);
        Assert.Empty(workerGroup.Messages);
    }

    [Fact]
    public void WorkerPropagatesForwardFailureToItsHost()
    {
        var group = new ControlGroup(1);
        group.Messages.Enqueue((1, [1017]));
        var failure = new InvalidOperationException("native collective failed");
        using var worker = new ControlModel(_path, group) { ForwardFailure = failure };
        Assert.Same(failure, Assert.Throws<InvalidOperationException>(() => worker.RunDistributedWorkerLoop()));
    }

    [Fact]
    public void RefusedTruncationDoesNotRewindWorkers()
    {
        var group = new ControlGroup(0);
        using var driver = new ControlModel(_path, group) { TruncateAllowed = false };
        driver.BeginDistributedDriver();
        Assert.False(driver.TryTruncateKVCache(4));
        Assert.Empty(group.Messages);
        Assert.Empty(driver.Executed);
        driver.TruncateAllowed = true;
        Assert.True(driver.TryTruncateKVCache(4));
        Assert.Equal(5, Assert.Single(group.Messages).op);
        Assert.Equal(new[] { "truncate:4" }, driver.Executed);
    }

    [Fact]
    public void FailedTruncationAbortsTheClusterAndPreventsSubsequentForward()
    {
        var group = new ControlGroup(0);
        var failure = new InvalidOperationException("truncate failed after modifying local cache");
        using var driver = new ControlModel(_path, group) { TruncateFailure = failure };
        driver.BeginDistributedDriver();
        Assert.Same(failure, Assert.Throws<InvalidOperationException>(() => driver.TryTruncateKVCache(4)));
        Assert.Same(failure, group.AbortFailure);
        Assert.Empty(group.Messages);
        Assert.Throws<InvalidOperationException>(() => driver.Forward([1017]));
        Assert.Equal(new[] { "truncate:4" }, driver.Executed);
        Assert.Empty(group.Messages);
    }

    [Fact]
    public void WorkerPropagatesProtocolFailureAndDriverCannotEnterWorkerLoop()
    {
        var group = new ControlGroup(1);
        group.Messages.Enqueue((999, []));
        using var worker = new ControlModel(_path, group);
        Assert.Contains("Unknown TP control op", Assert.Throws<InvalidOperationException>(worker.RunDistributedWorkerLoop).Message);
        using var driver = new ControlModel(_path, new ControlGroup(0));
        Assert.Contains("only run on a worker", Assert.Throws<InvalidOperationException>(driver.RunDistributedWorkerLoop).Message);
    }

    [Fact]
    public void DistributedBackendRestrictionsAreCheckedBeforeModelConstruction()
    {
        var descriptor = new ModelArchitectureDescriptor
        {
            Id = "native-distributed-probe",
            Aliases = ["native-distributed-probe"],
            Factory = _ => throw new InvalidOperationException("must not construct a model during preflight"),
            Serving = ServingContract.SharedRuntime(hasRoutedExperts: false),
            DistributedTensorParallelBackends = [BackendType.GgmlCuda],
        };
        ModelBase.ValidateDistributedTensorParallelism(descriptor, BackendType.GgmlCuda);
        foreach ((var backend, string option) in new[] { (BackendType.Cuda, "cuda"), (BackendType.GgmlVulkan, "ggml_vulkan") })
        {
            var error = Assert.Throws<NotSupportedException>(() => ModelBase.ValidateDistributedTensorParallelism(descriptor, backend));
            Assert.Contains("native-distributed-probe", error.Message);
            // Both backends in the spelling --backend accepts, not the enum names.
            Assert.Contains($"on --backend {option}.", error.Message);
            Assert.Contains("--backend ggml_cuda", error.Message);
        }
    }

    private sealed class ControlModel(string path, ITensorParallelGroup group) : ModelBase(path, BackendType.Cpu, 1, group)
    {
        public List<string> Executed { get; } = [];
        public Exception ForwardFailure { get; init; }
        public Exception TruncateFailure { get; init; }
        public bool TruncateAllowed { get; set; } = true;
        public override bool SupportsKVCacheTruncation => true;
        protected override float[] ForwardCore(int[] tokens)
        {
            if (ForwardFailure != null) throw ForwardFailure;
            Executed.Add("forward:" + string.Join(",", tokens));
            return [1f];
        }
        protected override float[] ForwardRefillCore(int[] tokens)
        {
            Executed.Add("refill:" + string.Join(",", tokens));
            return [1f];
        }
        protected override void ResetKVCacheCore() => Executed.Add("reset");
        protected override void TruncateKVCacheCore(int tokenCount)
        {
            Executed.Add("truncate:" + tokenCount);
            if (TruncateFailure != null) throw TruncateFailure;
        }
        protected override bool TryTruncateKVCacheCore(int tokenCount)
        {
            if (!TruncateAllowed) return false;
            TruncateKVCacheCore(tokenCount);
            return true;
        }
    }

    private sealed class ControlGroup(int rankOffset) : ITensorParallelGroup, IDistributedExecutionGroup
    {
        public Exception AbortFailure { get; private set; }
        public void Abort(Exception failure) => AbortFailure = failure;
        public Queue<(int op, int[] payload)> Messages { get; } = [];
        public int Degree => 1;
        public bool IsActive => true;
        public int GlobalDegree => 2;
        public int GlobalRankOffset => rankOffset;
        public int NodeCount => 2;
        public IAllocator GetAllocator(int rank) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => Messages.Enqueue((op, (int[])payload.Clone()));
        public (int op, int[] payload) ReceiveControl() => Messages.Dequeue();
        public void Dispose() { }
    }
}
