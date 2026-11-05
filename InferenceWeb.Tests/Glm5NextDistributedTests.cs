// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Net;
using System.Net.Sockets;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Distributed;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>Real TCP and native CPU graphs exercise the production stage and
/// rank boundaries. These fixtures are numerical/state tests, not GPU benchmarks.</summary>
public sealed class Glm5NextDistributedTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), $"glm-distributed-{Guid.NewGuid():N}");
    public Glm5NextDistributedTests() => Directory.CreateDirectory(_directory);
    public void Dispose() => Directory.Delete(_directory, recursive: true);

    private string Fixture(string name, string architecture = "glm5next", int heads = 8, bool tiedOutput = false, int expertFfn = 128) =>
        GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(Path.Combine(_directory, name + ".gguf"),
            heads, quantizeAttentionOutput: true, numLayers: 6, mixedAttention: true,
            routedExperts: true, quantizeExperts: true, expertFfn: expertFfn, architecture: architecture, tiedOutput: tiedOutput);

    private static NativeEnvScope EnvironmentForFixture()
    {
        var environment = new NativeEnvScope();
        environment.Set("TS_GLM_TP_OVERSUBSCRIBE", "1");
        environment.Set("MAX_CONTEXT", "256");
        environment.Set("TS_GLM_UBATCH", "4");
        // One CPU worker per rank: every rank of these fixtures lives in this process. Without
        // a GPU the native loader sizes each rank's CPU pool to every usable core and only
        // TS_CPU_MOE_THREADS overrides that (TS_GLM_THREADS does not), so on an 18-core Mac two
        // ranks spun 36 workers in ggml's barriers: 9-21 s warm-ups and the 45 s timeouts.
        environment.Set("TS_GLM_THREADS", null);
        environment.Set("TS_CPU_MOE_THREADS", "1");
        environment.Set("TS_GLM_NATIVE", null);
        environment.Set("TS_GLM_TP_SHARD", null);
        environment.Set("TS_GLM_TRACE", null);
        environment.Set("TS_SPEC", "0");
        return environment;
    }

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(true, 2, 1, "glm5next")]
    [InlineData(true, 2, 2, "glm5-next")]
    [InlineData(true, 3, 1, "glm5-next")]
    [InlineData(false, 2, 1, "glm5-next")]
    [InlineData(false, 2, 2, "glm5next")]
    // Distributed TP shards routed-expert channels, while each rank keeps
    // complete attention: fewer attention heads than global ranks is valid.
    [InlineData(false, 2, 2, "glm5next", false, 2)]
    // The down dot keeps its complete input. Its single Q8 block may feed
    // four logical activation shards; the old input-block rank limit is gone.
    [InlineData(false, 2, 2, "glm5next", false, 2, 32)]
    [InlineData(true, 2, 2, "glm5-next", true)]
    public async Task NativeMixedAttentionMatchesLocalAcrossRefillDecodeAndReset(
        bool layerSplit, int nodeCount, int localDegree, string architecture, bool tiedOutput = false, int heads = 8, int expertFfn = 128)
    {
        using var environment = EnvironmentForFixture();
        string path = Fixture("mixed", architecture, heads: heads, tiedOutput: tiedOutput, expertFfn: expertFfn);
        using var reference = new GlmDsaModel(path, BackendType.GgmlCpu);
        var endpoints = Endpoints(nodeCount);
        var groups = new ITensorParallelGroup[nodeCount];
        GlmDsaModel[] models = [];
        Task[] workers = [];
        try
        {
            models = await Task.WhenAll(Enumerable.Range(0, nodeCount).Select(node => Task.Run(() =>
            {
                ITensorParallelGroup group = layerSplit
                    ? new DistributedLayerSplitGroup(new CpuDevices(localDegree), node, endpoints)
                    : new DistributedTensorParallelGroup(new CpuDevices(localDegree), node, endpoints);
                groups[node] = group;
                return new GlmDsaModel(path, BackendType.GgmlCpu,
                    tpDegree: layerSplit ? 1 : localDegree, tpGroup: group,
                    // A supplied group also defines placement when the public
                    // optional degree is left at its default.
                    layerSplitDegree: layerSplit && !tiedOutput ? localDegree : 1);
            }))).WaitAsync(TimeSpan.FromSeconds(45));

            Assert.All(models, model =>
            {
                Assert.Equal("glm5next", model.Config.Architecture);
                Assert.Equal(heads, model.Config.NumHeads);
                Assert.False(model.SupportsKVCacheTruncation);
                Assert.False(model.SupportsPerSequenceFusedForward);
                Assert.False(model.SpeculationProfitable);
                Assert.Throws<NotSupportedException>(model.SpecSnapshotRecurrentState);
                Assert.False(model.GetPrefixCacheCapabilities().AdoptPrimaryOnDisplacement);
                Assert.False(model.TryConvertPrimary("unmirrored", 0, out _));
                Assert.Throws<NotSupportedException>(() => model.AdoptPrimaryCacheToFused("unmirrored"));
                // The public factory checks this after construction. Native
                // sharding must not be mistaken for empty managed shard maps.
                typeof(ModelBase).GetMethod("VerifyTensorParallelShardedWeights",
                    System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
                    .Invoke(model, [model.Config.Architecture]);
            });
            // Warmup runs on all nodes before the driver's control loop starts.
            await Task.WhenAll(models.Select(model => Task.Run(model.WarmUpKernels)))
                .WaitAsync(TimeSpan.FromSeconds(45));
            var driver = models[0];
            workers = models.Skip(1).Select(model => Task.Run(model.RunDistributedWorkerLoop)).ToArray();
            driver.BeginDistributedDriver();
            async Task Compare(int[] tokens, bool refill)
            {
                float[] expected = (float[])(refill ? reference.ForwardRefill(tokens) : reference.Forward(tokens)).Clone();
                float[] actual = await Task.Run(() => (float[])(refill ? driver.ForwardRefill(tokens) : driver.Forward(tokens)).Clone())
                    .WaitAsync(TimeSpan.FromSeconds(45));
                Assert.Equal(128, actual.Length);
                Assert.Equal(ArgMax(expected), ArgMax(actual));
                for (int i = 0; i < actual.Length; i++)
                {
                    Assert.True(float.IsFinite(actual[i]) && float.IsFinite(expected[i]));
                    Assert.InRange(MathF.Abs(actual[i] - expected[i]), 0, layerSplit ? 1e-6f : 2e-4f);
                }
                Assert.Equal(reference.CacheSeqLen, driver.CacheSeqLen);
            }
            // Pooled top-k=4 with pool width=4 is active after the first ubatch.
            await Compare(Enumerable.Range(0, 25).Select(i => 40 + i * 7 % 70).ToArray(), true);
            await Compare([65], false);
            await Compare([66], false);
            await Compare([67, 68, 69, 70, 71], true);
            reference.ResetKVCache();
            driver.ResetKVCache();
            await Compare([72, 73, 74, 75, 76, 77, 78, 79, 80], true);
            await Compare([81], false);
            groups[0].BroadcastControl(4, []);
            await Task.WhenAll(workers).WaitAsync(TimeSpan.FromSeconds(15));
            Assert.All(models, model => Assert.Equal(reference.CacheSeqLen, model.CacheSeqLen));
        }
        finally
        {
            foreach (var group in groups) (group as IDistributedExecutionGroup)?.Abort(new OperationCanceledException("Fixture complete"));
            foreach (var model in models) model.Dispose();
            foreach (var group in groups) group?.Dispose();
            try { await Task.WhenAll(workers).WaitAsync(TimeSpan.FromSeconds(10)); } catch { }
        }
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public async Task ExplicitDegreeMismatchAbortsPeerBeforeBaseConstruction()
    {
        using var environment = EnvironmentForFixture();
        string path = Fixture("degree");
        var endpoints = Endpoints(2);
        var groups = await Task.WhenAll(Enumerable.Range(0, 2).Select(node => Task.Run(() =>
            new DistributedLayerSplitGroup(new CpuDevices(2), node, endpoints)))).WaitAsync(TimeSpan.FromSeconds(20));
        try
        {
            var outcomes = await Task.WhenAll(Enumerable.Range(0, 2).Select(node => Task.Run(() =>
                Record.Exception(() => { using var model = new GlmDsaModel(path, BackendType.GgmlCpu,
                    tpGroup: groups[node], layerSplitDegree: node == 0 ? 3 : 2); }))))
                .WaitAsync(TimeSpan.FromSeconds(20));
            Assert.Contains("degree disagrees", Assert.IsType<ArgumentException>(outcomes[0]).Message);
            Assert.NotNull(outcomes[1]);
        }
        finally { foreach (var group in groups) group.Dispose(); }
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public async Task DifferentModelMetadataAbortsBothNodesBeforeWeightsLoad()
    {
        using var environment = EnvironmentForFixture();
        string[] paths = [Fixture("first", heads: 8), Fixture("second", heads: 12)];
        var endpoints = Endpoints(2);
        var outcomes = await Task.WhenAll(Enumerable.Range(0, 2).Select(node => Task.Run(() =>
        {
            using var group = new DistributedLayerSplitGroup(new CpuDevices(1), node, endpoints);
            return Record.Exception(() =>
            {
                using var model = new GlmDsaModel(paths[node], BackendType.GgmlCpu, tpGroup: group);
            });
        }))).WaitAsync(TimeSpan.FromSeconds(30));
        Assert.All(outcomes, error => Assert.Contains("different model metadata", Assert.IsType<InvalidOperationException>(error).Message));
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public async Task MalformedMetadataReleasesMappingAndAbortsWaitingPeer()
    {
        using var environment = EnvironmentForFixture();
        string invalid = Path.Combine(_directory, "invalid.gguf");
        BackendFailureWarmupTests.WriteProbeGguf(invalid); // structurally valid GGUF, no model dimensions
        string[] paths = [invalid, Fixture("valid")];
        var endpoints = Endpoints(2);
        var local = new[] { new CpuDevices(1), new CpuDevices(1) };
        var groups = await Task.WhenAll(Enumerable.Range(0, 2).Select(node => Task.Run(() =>
            new DistributedLayerSplitGroup(local[node], node, endpoints)))).WaitAsync(TimeSpan.FromSeconds(20));
        try
        {
            var outcomes = await Task.WhenAll(Enumerable.Range(0, 2).Select(node => Task.Run(() =>
                Record.Exception(() => { using var model = new GlmDsaModel(paths[node], BackendType.GgmlCpu, tpGroup: groups[node]); }))))
                .WaitAsync(TimeSpan.FromSeconds(20));
            Assert.All(outcomes, Assert.NotNull);
            Assert.Contains("nextn_predict_layers", outcomes[0].Message);
            Assert.All(local, devices => Assert.True(devices.Disposed));
            // A failed constructor must release its GGUF mapping, not rely on GC.
            using var exclusive = new FileStream(invalid, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        }
        finally { foreach (var group in groups) group.Dispose(); }
    }

    private static int ArgMax(float[] values) => Array.IndexOf(values, values.Max());

    private sealed class CpuDevices(int degree) : ITensorParallelGroup
    {
        private readonly CpuAllocator[] _allocators = Enumerable.Range(0, degree).Select(_ => new CpuAllocator(BlasEnum.DotNet)).ToArray();
        public int Degree => degree;
        public int GlobalDegree => degree;
        public int GlobalRankOffset => 0;
        public int NodeCount => 1;
        public bool IsActive => false;
        public bool Disposed { get; private set; }
        public IAllocator GetAllocator(int rank) => _allocators[rank];
        public void AllReduce(Tensor[] tensors) => throw new InvalidOperationException("Native GLM owns local rank reductions.");
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() => Disposed = true;
    }

    private static IPEndPoint[] Endpoints(int count)
    {
        var listeners = Enumerable.Range(0, count).Select(_ => new TcpListener(IPAddress.Loopback, 0)).ToArray();
        try
        {
            foreach (var listener in listeners) listener.Start();
            return listeners.Select(listener => (IPEndPoint)listener.LocalEndpoint).ToArray();
        }
        finally { foreach (var listener in listeners) listener.Stop(); }
    }
}
