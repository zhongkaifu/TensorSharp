// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace InferenceWeb.Tests;

public sealed class DeepSeek41DistributedTests : IDisposable
{
    private readonly EnvScope _env = new();

    public DeepSeek41DistributedTests()
    {
        _env.ClearSpeculationVars();
        _env.Set("KV_CACHE_DTYPE", "f16");
    }

    public void Dispose() => _env.Dispose();

    [Theory]
    [InlineData(1, 2, 0, 2)]
    [InlineData(1, 2, 1, 2)]
    [InlineData(2, 4, 0, 2)]
    [InlineData(2, 4, 2, 2)]
    [InlineData(3, 6, 3, 2)]
    [InlineData(8, 16, 8, 2)]
    public void ExistingNestedTransportSupportsEveryLocalRankInTheGlobalInterval(int local, int global, int offset, int nodes)
    {
        var group = new Group(local, global, offset, nodes);
        DeepSeek41Architecture.ValidateTensorParallelGroup(BackendType.GgmlCuda, local, group);
        DeepSeek41Architecture.ValidateLoad("not-opened.gguf", BackendType.GgmlCuda, null, group, local);
        ITensorParallelGroup resolved = group;
        Assert.Equal(local, ModelBase.ResolveTensorParallelSupport(DeepSeek41Architecture.Descriptor,
            BackendType.GgmlCuda, local, ref resolved, out int split));
        Assert.Same(group, resolved);
        Assert.Equal(1, split);
    }

    [Theory]
    [InlineData(2, 2, 3, 0, 2)]
    [InlineData(2, 2, 4, 1, 2)]
    [InlineData(2, 2, 4, 4, 2)]
    [InlineData(2, 2, 4, -2, 2)]
    [InlineData(2, 2, 18, 0, 9)]
    [InlineData(2, 2, 2, 0, 2)]
    [InlineData(1, 1, 1, 0, 2)]
    [InlineData(2, 1, 2, 0, 2)]
    public void IncompleteOrOverlappingTopologyFailsBeforeWeights(int requested, int local, int global, int offset, int nodes)
        => Assert.Throws<ArgumentException>(() => DeepSeek41Architecture.ValidateTensorParallelGroup(
            BackendType.GgmlCuda, requested, new Group(local, global, offset, nodes)));

    [Theory]
    [InlineData(BackendType.Cuda)]
    [InlineData(BackendType.Cpu)]
    [InlineData(BackendType.GgmlCpu)]
    [InlineData(BackendType.GgmlVulkan)]
    [InlineData(BackendType.GgmlMetal)]
    public void UnsupportedDistributedBackendsAreRejectedBeforeConnecting(BackendType backend)
    {
        Assert.Throws<NotSupportedException>(() => ModelBase.ValidateDistributedTensorParallelism(
            DeepSeek41Architecture.Descriptor, backend));
        Assert.Throws<NotSupportedException>(() => DeepSeek41Architecture.ValidateTensorParallelGroup(
            backend, 2, new Group(2, 4, 0, 2)));
    }

    [Fact]
    public void GgmlCudaPassesDistributedPreflightAndPreservesLocalLayerPlacement()
    {
        ModelBase.ValidateDistributedTensorParallelism(DeepSeek41Architecture.Descriptor, BackendType.GgmlCuda);
        Assert.True(DeepSeek41Architecture.Descriptor.SupportsDistributedTensorParallel);
        Assert.True(DeepSeek41Architecture.Descriptor.Serving.TensorParallel.IsSupportedOn(BackendType.GgmlCuda));
        ITensorParallelGroup group = null;
        Assert.Equal(1, ModelBase.ResolveTensorParallelSupport(DeepSeek41Architecture.Descriptor,
            BackendType.GgmlCuda, 1, ref group, out int split, 2));
        Assert.Equal(2, split);
        Assert.Null(group);
    }

    [Fact]
    public void DistributedRequestCannotCombineLayerSplitOrUnmirroredDraftOperations()
    {
        ITensorParallelGroup group = new Group(2, 4, 0, 2);
        Assert.Throws<NotSupportedException>(() => ModelBase.ResolveTensorParallelSupport(DeepSeek41Architecture.Descriptor,
            BackendType.GgmlCuda, 2, ref group, out _, 2));
        var error = Assert.Throws<NotSupportedException>(() => DeepSeek41Architecture.ValidateLoad(
            "not-opened.gguf", BackendType.GgmlCuda, "not-opened-draft.gguf", group, 2));
        Assert.Contains("DSpark", error.Message);
    }

    [Fact]
    public void MatchingConfigurationVotesPassAndUseTheExistingCollective()
    {
        int calls = 0;
        var group = new Group(2, 4, 0, 2, (buffer, count) =>
        {
            Assert.Equal(256, count);
            Assert.All(buffer, value => Assert.True(value is 0 or 1));
            for (int i = 0; i < count; i++) buffer[i] *= 2;
            calls++;
        });
        DeepSeek41Architecture.ValidateDistributedExecutionConfig(group, "identical-checkpoint-and-settings");
        Assert.Equal(1, calls);
    }

    [Theory]
    [InlineData(1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void DifferentOrNonfiniteConfigurationVotesFailBeforeNativeLoad(float invalidVote)
    {
        var group = new Group(2, 4, 0, 2, (buffer, count) =>
        {
            for (int i = 0; i < count; i++) buffer[i] *= 2;
            buffer[0] = invalidVote;
        });
        Assert.Throws<InvalidOperationException>(() =>
            DeepSeek41Architecture.ValidateDistributedExecutionConfig(group, "same-local-configuration"));
    }

    [Fact]
    public void ConfigurationTransportFailureIsNotAValidationPass()
    {
        var group = new Group(2, 4, 0, 2, (_, _) => throw new IOException("transport unavailable"));
        Assert.Throws<IOException>(() => DeepSeek41Architecture.ValidateDistributedExecutionConfig(group, "settings"));
        DeepSeek41Architecture.ValidateDistributedExecutionConfig(null, "no-group");
        DeepSeek41Architecture.ValidateDistributedExecutionConfig(new Group(2, 2, 0, 1), "local-only");
    }

    [Fact]
    public void FailureBeforeTheNativeHookClosesTransportAndRefusesFurtherForwardOrRefill()
    {
        // Deliberately omit Config to inject an allocation/setup-side failure
        // before any native call. No model files, GPU or sockets are created.
        var model = (DeepSeek4Model)RuntimeHelpers.GetUninitializedObject(typeof(DeepSeek4Model));
        var group = new Group(2, 4, 0, 2);
        const BindingFlags instance = BindingFlags.Instance | BindingFlags.NonPublic;
        typeof(DeepSeek4Model).GetField("_sync", instance)!.SetValue(model, new object());
        typeof(ModelBase).GetField("_tpGroup", instance)!.SetValue(model, group);
        MethodInfo forward = typeof(DeepSeek4Model).GetMethod("ForwardCore", instance)!;
        var original = Assert.Throws<TargetInvocationException>(() => forward.Invoke(model, new object[] { new[] { 1 } }));
        Assert.IsType<NullReferenceException>(original.InnerException);
        Assert.Equal(1, group.DisposeCount);
        foreach (MethodInfo operation in new[] { forward, typeof(ModelBase).GetMethod("ForwardRefillCore", instance)! })
        {
            var repeated = Assert.Throws<TargetInvocationException>(() => operation.Invoke(model, new object[] { new[] { 1 } }));
            var error = Assert.IsType<InvalidOperationException>(repeated.InnerException);
            Assert.Contains("restart every node", error.Message);
            Assert.Equal(1, group.DisposeCount);
        }
    }

    [Theory]
    [InlineData(1, 2, 0)]
    [InlineData(2, 2, 1)]
    [InlineData(3, 3, 2)]
    public void LayerPipelineUsesItsOwnPlacementWithoutTensorParallelSharding(int local, int nodes, int node)
    {
        var layers = new LayerGroup(local, nodes, node);
        DeepSeek41Architecture.ValidateTensorParallelGroup(BackendType.GgmlCuda, 1, layers);
        DeepSeek41Architecture.ValidateLoad("not-opened.gguf", BackendType.GgmlCuda, null, layers, 1);
        ITensorParallelGroup group = layers;
        Assert.Equal(1, ModelBase.ResolveTensorParallelSupport(DeepSeek41Architecture.Descriptor,
            BackendType.GgmlCuda, 1, ref group, out int split, local));
        Assert.Equal(local, split);
        Assert.Same(layers, group);
        // The direct-CUDA engine runs whole-layer stages too (Dsv4CudaEngine.Distributed.cs).
        DeepSeek41Architecture.ValidateTensorParallelGroup(BackendType.Cuda, 1, layers);
        DeepSeek41Architecture.ValidateLoad("not-opened.gguf", BackendType.Cuda, null, layers, 1);
        group = layers;
        Assert.Equal(1, ModelBase.ResolveTensorParallelSupport(DeepSeek41Architecture.Descriptor,
            BackendType.Cuda, 1, ref group, out split, local));
        Assert.Equal(local, split);
        foreach (BackendType other in new[] { BackendType.GgmlVulkan, BackendType.GgmlCpu, BackendType.Cpu })
            Assert.Throws<NotSupportedException>(() => DeepSeek41Architecture.ValidateTensorParallelGroup(other, 1, layers));
        Assert.Throws<NotSupportedException>(() =>
            DeepSeek41Architecture.ValidateTensorParallelGroup(BackendType.GgmlCuda, 2, layers));
        Assert.Throws<NotSupportedException>(() =>
            DeepSeek41Architecture.ValidateTensorParallelGroup(BackendType.Cuda, 2, layers));
    }

    /// <summary>
    /// --backend cuda runs a distributed layer split of V4.1 but not distributed (or local) tensor parallelism:
    /// that would need 80 cross-node collectives per token and both Engram tables on every node.
    /// </summary>
    [Fact]
    public void CudaRunsDistributedLayerSplitButNotTensorParallelism()
    {
        Assert.Equal(new[] { BackendType.GgmlCuda, BackendType.Cuda }, DeepSeek41Architecture.Descriptor.DistributedLayerSplitBackends);
        ModelBase.ValidateDistributedLayerSplit(DeepSeek41Architecture.Descriptor, BackendType.Cuda);
        var distributedTp = Assert.Throws<NotSupportedException>(() =>
            ModelBase.ValidateDistributedTensorParallelism(DeepSeek41Architecture.Descriptor, BackendType.Cuda));
        Assert.Contains("--backend ggml_cuda", distributedTp.Message);
        var group = Assert.Throws<NotSupportedException>(() =>
            DeepSeek41Architecture.ValidateTensorParallelGroup(BackendType.Cuda, 2, new Group(2, 4, 0, 2)));
        Assert.Contains("requires --backend ggml_cuda", group.Message);
        var local = Assert.Throws<NotSupportedException>(() =>
            DeepSeek41Architecture.ValidateLoad("not-opened.gguf", BackendType.Cuda, null, null, 2));
        Assert.Contains("ggml_cuda", local.Message);
    }

    /// <summary>The distributed stages do not mirror DSpark's drafting and rewinds: a drafter with a cuda (or
    /// ggml_cuda) layer split is refused before anything is opened.</summary>
    [Theory]
    [InlineData(BackendType.Cuda)]
    [InlineData(BackendType.GgmlCuda)]
    public void DsparkIsRefusedUnderADistributedLayerSplit(BackendType backend)
    {
        var layers = new LayerGroup(1, 2, 0);
        var error = Assert.Throws<NotSupportedException>(() => DeepSeek41Architecture.ValidateLoad(
            "not-opened.gguf", backend, "not-opened-draft.gguf", layers, 1));
        Assert.Contains("DSpark", error.Message);
        Assert.Contains("layer split", error.Message);
    }

    [Fact]
    public void LayerFailureAbortsTransportWithoutFreeingLiveDeviceAllocations()
    {
        var model = (DeepSeek4Model)RuntimeHelpers.GetUninitializedObject(typeof(DeepSeek4Model));
        var layers = new LayerGroup(2, 2, 0);
        const BindingFlags instance = BindingFlags.Instance | BindingFlags.NonPublic;
        typeof(DeepSeek4Model).GetField("_sync", instance)!.SetValue(model, new object());
        typeof(ModelBase).GetField("<LayerSplitGroup>k__BackingField", instance)!.SetValue(model, layers);
        MethodInfo forward = typeof(DeepSeek4Model).GetMethod("ForwardCore", instance)!;
        var original = Assert.Throws<TargetInvocationException>(() => forward.Invoke(model, new object[] { new[] { 1 } }));
        Assert.IsType<NullReferenceException>(original.InnerException);
        Assert.Equal(1, layers.AbortCount);
        Assert.Same(original.InnerException, layers.Failure);
        Assert.Equal(0, layers.DisposeCount);
        var reset = typeof(DeepSeek4Model).GetMethod("ResetKVCacheCore", instance)!;
        var repeated = Assert.Throws<TargetInvocationException>(() => reset.Invoke(model, null));
        Assert.Contains("restart every node", Assert.IsType<InvalidOperationException>(repeated.InnerException).Message);
        Assert.Equal(1, layers.AbortCount);
        Assert.Equal(0, layers.DisposeCount);
    }

    private sealed class LayerGroup(int local, int nodes, int node)
        : Group(local, checked(local * nodes), checked(local * node), nodes), ILayerSplitGroup, IDistributedExecutionGroup
    {
        public int NodeId => node;
        public int AbortCount { get; private set; }
        public Exception? Failure { get; private set; }
        public void Abort(Exception failure) { AbortCount++; Failure = failure; }
        public void Send(float[] buffer, int count, int destination) => throw new NotSupportedException();
        public void Receive(float[] buffer, int count, int source) => throw new NotSupportedException();
    }

    private class Group(int local, int global, int offset, int nodes, Action<float[], int>? reduce = null)
        : ITensorParallelGroup, INestedTensorParallelGroup
    {
        public int Degree => local;
        public int GlobalDegree => global;
        public int GlobalRankOffset => offset;
        public int NodeCount => nodes;
        public bool IsActive => true;
        public int DisposeCount { get; private set; }
        public ITensorParallelGroup LocalGroup => throw new NotSupportedException("Geometry validation must not initialize devices.");
        public IAllocator GetAllocator(int rank) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void CrossNodeAllReduce(float[] buffer, int count)
        {
            if (reduce == null) throw new NotSupportedException();
            reduce(buffer, count);
        }
        public void Synchronize() => throw new NotSupportedException();
        public void Barrier() => throw new NotSupportedException();
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() => DisposeCount++;
    }
}
