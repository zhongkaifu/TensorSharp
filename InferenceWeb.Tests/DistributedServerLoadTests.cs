// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Server;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public sealed class DistributedServerLoadTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"distributed-server-{Guid.NewGuid():N}.gguf");
    private readonly string _savedNode = Environment.GetEnvironmentVariable("TENSORSHARP_TP_NODE_ID");
    private readonly string _savedPeers = Environment.GetEnvironmentVariable("TENSORSHARP_TP_PEERS");

    public DistributedServerLoadTests()
    {
        BackendFailureWarmupTests.WriteProbeGguf(_path);
        Environment.SetEnvironmentVariable("TENSORSHARP_TP_NODE_ID", null);
        Environment.SetEnvironmentVariable("TENSORSHARP_TP_PEERS", null);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("TENSORSHARP_TP_NODE_ID", _savedNode);
        Environment.SetEnvironmentVariable("TENSORSHARP_TP_PEERS", _savedPeers);
        File.Delete(_path);
    }

    [Fact]
    public void RuntimeDistributedLoadsAreRefusedBeforeConnectingOrReplacingTheModel()
    {
        using var service = new ModelService(null, (path, _, _, _) => new ProbeModel(path));
        service.LoadModel(_path, null, "cpu");
        var original = service.Model;
        service.TensorParallelGroupFactory = _ => throw new InvalidOperationException("must not connect");
        EnableDistributed();
        service.LoadModel(_path, null, "cpu");
        Assert.Same(original, service.Model);
        foreach (var request in new[] { (Path: _path + ".missing", Backend: "cpu"), (Path: _path, Backend: "ggml_cuda") })
        {
            var error = Assert.Throws<ModelLoadRefusedException>(() => service.LoadModel(request.Path, null, request.Backend));
            Assert.Contains("startup --model", error.Message);
            Assert.Same(original, service.Model);
        }
    }

    [Fact]
    public void InitialDistributedLoadRequiresTheStartupRoute()
    {
        using var lifecycle = new ModelLifecycleService(null, (_, _, _, _) => throw new InvalidOperationException("must not load"));
        lifecycle.TensorParallelGroupFactory = _ => throw new InvalidOperationException("must not connect");
        EnableDistributed();
        Assert.Throws<ModelLoadRefusedException>(() => lifecycle.ShouldLoadDistributedModel(_path, null, "ggml_cuda", startupLoad: false));
        Assert.True(lifecycle.ShouldLoadDistributedModel(_path, null, "ggml_cuda", startupLoad: true));
    }

    [Fact]
    public void DefaultServerFactoryPassesTheSuppliedGroupsLocalDegreeToTheArchitecture()
    {
        string architecture = "distributed-server-degree-" + Guid.NewGuid().ToString("N");
        ModelCreateContext observed = null;
        var stop = new InvalidOperationException("Captured context before native model construction");
        ModelArchitectureRegistry.Register(new ModelArchitectureDescriptor
        {
            Id = architecture,
            Aliases = [architecture],
            ApplyNativeTunables = context => { observed = context; throw stop; },
            Factory = _ => throw new InvalidOperationException("must not construct a native model"),
            Serving = ServingContract.SharedRuntime(hasRoutedExperts: false),
        });
        SyntheticGguf.Write(_path, [new SyntheticGguf.Str { Key = "general.architecture", V = architecture }], []);
        var group = new DegreeProbeGroup();
        using var lifecycle = new ModelLifecycleService(null);
        lifecycle.TensorParallelGroupFactory = _ => group;
        Assert.Same(stop, Assert.Throws<InvalidOperationException>(() => lifecycle.LoadModel(_path, null, "ggml_cuda")));
        Assert.NotNull(observed);
        Assert.Equal(2, observed.TpDegree);
        Assert.Same(group, observed.TpGroup);
        Assert.True(group.Disposed);
    }

    private static void EnableDistributed()
    {
        Environment.SetEnvironmentVariable("TENSORSHARP_TP_NODE_ID", "0");
        Environment.SetEnvironmentVariable("TENSORSHARP_TP_PEERS", "127.0.0.1:9500,127.0.0.1:9501");
    }

    [Fact]
    public void DefaultServerFactoryPreservesLayerModeAndLocalDegree()
    {
        string architecture = "distributed-server-layer-" + Guid.NewGuid().ToString("N");
        ModelCreateContext observed = null;
        var stop = new InvalidOperationException("Captured layer context before native construction");
        ModelArchitectureRegistry.Register(new ModelArchitectureDescriptor
        {
            Id = architecture,
            Aliases = [architecture],
            Serving = ServingContract.SharedRuntime(hasRoutedExperts: false),
            SupportsDistributedLayerSplit = true,
            ApplyNativeTunables = context => { observed = context; throw stop; },
            Factory = _ => throw new InvalidOperationException("must not construct"),
        });
        SyntheticGguf.Write(_path, [new SyntheticGguf.Str { Key = "general.architecture", V = architecture }], []);
        var group = new LayerProbeGroup();
        using var lifecycle = new ModelLifecycleService(null);
        lifecycle.TensorParallelGroupFactory = _ => group;
        Assert.Same(stop, Assert.Throws<InvalidOperationException>(() => lifecycle.LoadModel(_path, null, "ggml_cuda")));
        Assert.Equal(1, observed.TpDegree);
        Assert.Equal(2, observed.LayerSplitDegree);
        Assert.Same(group, observed.TpGroup);
        Assert.True(group.Disposed);
    }

    /// <summary>A node whose two GPUs are one tensor-parallel stage of a multi-node layer split (hybrid) loads as one
    /// local stage, as the CLI's <c>--tp 2 --layer-split 1</c> does: the server passed its GPU count as the layer-split
    /// degree and the load refused it.</summary>
    [Fact]
    public void DefaultServerFactoryLoadsAHybridNodeAsItsLocalStages()
    {
        string architecture = "distributed-server-hybrid-" + Guid.NewGuid().ToString("N");
        ModelCreateContext observed = null;
        var stop = new InvalidOperationException("Captured hybrid context before native construction");
        ModelArchitectureRegistry.Register(new ModelArchitectureDescriptor
        {
            Id = architecture,
            Aliases = [architecture],
            Serving = ServingContract.SharedRuntime(hasRoutedExperts: false),
            SupportsDistributedLayerSplit = true,
            ApplyNativeTunables = context => { observed = context; throw stop; },
            Factory = _ => throw new InvalidOperationException("must not construct"),
        });
        SyntheticGguf.Write(_path, [new SyntheticGguf.Str { Key = "general.architecture", V = architecture }], []);
        var group = new HybridProbeGroup();
        using var lifecycle = new ModelLifecycleService(null);
        lifecycle.TensorParallelGroupFactory = _ => group;
        Assert.Same(stop, Assert.Throws<InvalidOperationException>(() => lifecycle.LoadModel(_path, null, "ggml_cuda")));
        Assert.Equal(1, observed.LayerSplitDegree);
        Assert.Same(group, observed.TpGroup);
    }

    private sealed class ProbeModel(string path) : ModelBase(path, BackendType.Cpu)
    {
        protected override float[] ForwardCore(int[] tokens) => [];
        protected override void ResetKVCacheCore() { }
    }

    private class DegreeProbeGroup : ITensorParallelGroup
    {
        public bool Disposed { get; private set; }
        public int Degree => 2;
        public bool IsActive => true;
        public int GlobalDegree => 4;
        public int GlobalRankOffset => 0;
        public int NodeCount => 2;
        public IAllocator GetAllocator(int rank) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() => Disposed = true;
    }

    private class LayerProbeGroup : DegreeProbeGroup, ILayerSplitGroup
    {
        public int NodeId => 0;
        public void Send(float[] buffer, int count, int destination) => throw new NotSupportedException();
        public void Receive(float[] buffer, int count, int source) => throw new NotSupportedException();
    }

    /// <summary>Two GPUs as one tensor-parallel stage.</summary>
    private sealed class HybridProbeGroup : LayerProbeGroup, ILayerSplitGroup
    {
        public int StageTensorParallel => 2;
    }
}
